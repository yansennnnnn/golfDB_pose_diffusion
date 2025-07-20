import torch
import torch.nn.functional as F
from typing import Dict, Any

from finetune.trainer import Trainer
from finetune.schemas import Components
from finetune.models.stage_recognizer import StageRecognizer
from finetune.models.keyframe_encoder import KeyframeEncoder


class GolfFinetuneTrainer(Trainer):
    """
    A specialized trainer for fine-tuning a video diffusion model for golf swing generation,
    based on the paper's structure-aware methodology.
    """

    def __init__(self, args):
        super().__init__(args)
        self.stage_loss_weight = 0.1 # Hyperparameter from the paper (alpha)

    def load_components(self) -> Components:
        """
        Loads the base model components and the new structure-aware components.
        """
        base_components = super().load_components()

        # Add our new models to the components bundle
        self.stage_recognizer = StageRecognizer(
            input_dim=256, 
            num_stages=8, # 8 stages for golf swing
            num_channels=[128, 128, 128]
        ).to(self.accelerator.device)

        self.keyframe_encoder = KeyframeEncoder(
            embedding_dim=self.text_encoder.config.hidden_size # Match text encoder dim
        ).to(self.accelerator.device)

        return base_components

    def prepare_trainable_parameters(self):
        """
        Set only the main model and our new components as trainable.
        """
        super().prepare_trainable_parameters()
        self.params_to_train += list(self.stage_recognizer.parameters())
        # Keyframe encoder uses a frozen ResNet, so only projection and transformer are trained
        self.params_to_train += list(self.keyframe_encoder.projection.parameters())
        self.params_to_train += list(self.keyframe_encoder.transformer_encoder.parameters())


    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Computes the combined loss, including diffusion loss and stage recognition loss.
        """
        # Unpack batch - assuming dataloader provides these keys
        video_frames = batch["video_frames"] # Target for diffusion model
        input_clip = batch["input_clip"] # Input for stage recognizer
        stage_labels = batch["stage_labels"] # Ground truth for stage recognizer
        keyframes = batch["keyframes"] # Input for keyframe encoder
        
        # 1. Standard Diffusion Loss with Keyframe Conditioning
        
        # Encode video frames into latent space
        latents = self.encode_video(video_frames)
        
        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Get keyframe embeddings to use as conditioning
        keyframe_embeds = self.keyframe_encoder(keyframes)
        
        # Get the target for loss depending on the prediction type
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        # The main model (transformer) needs to accept `encoder_hidden_states` for conditioning
        model_pred = self.transformer(noisy_latents, encoder_hidden_states=keyframe_embeds, timestep=timesteps).sample
        diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # 2. Auxiliary Stage Recognition Loss
        predicted_stages = self.stage_recognizer(input_clip)
        stage_loss = F.cross_entropy(predicted_stages, stage_labels)
        
        # 3. Combined Loss
        total_loss = diffusion_loss + self.stage_loss_weight * stage_loss
        
        # Log losses
        self.accelerator.log({
            "total_loss": total_loss.detach().item(),
            "diffusion_loss": diffusion_loss.detach().item(),
            "stage_loss": stage_loss.detach().item()
        })
        
        return total_loss 