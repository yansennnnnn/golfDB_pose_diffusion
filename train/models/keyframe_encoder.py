import torch
import torch.nn as nn
import torchvision.models as models

class KeyframeEncoder(nn.Module):
    """
    Encodes keyframe images into feature embeddings to be used as conditioning
    for the diffusion model. Uses a pre-trained ResNet and a small Transformer.
    """
    def __init__(self, embedding_dim, num_layers=2, nhead=4):
        super(KeyframeEncoder, self).__init__()
        # Use a pre-trained ResNet-18, remove the final classifier
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        resnet_output_dim = resnet.fc.in_features # 512 for ResNet-18
        
        # A small transformer to process the sequence of keyframe features
        self.projection = nn.Linear(resnet_output_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, keyframes):
        # keyframes shape: (B, Num_Keyframes, C, H, W)
        batch_size, num_keyframes, C, H, W = keyframes.shape
        
        # Reshape to process all keyframes at once
        keyframes_flat = keyframes.view(batch_size * num_keyframes, C, H, W)
        
        # Extract features
        features_flat = self.feature_extractor(keyframes_flat) # (B*N, 512, 1, 1)
        features = features_flat.view(batch_size, num_keyframes, -1) # (B, N, 512)
        
        # Project features to the embedding dimension
        projected_features = self.projection(features) # (B, N, embedding_dim)
        
        # Transformer expects (Seq_Len, Batch, Dim)
        projected_features = projected_features.permute(1, 0, 2)
        
        # Process through Transformer
        transformer_output = self.transformer_encoder(projected_features) # (N, B, embedding_dim)
        
        # Permute back to (B, N, embedding_dim)
        return transformer_output.permute(1, 0, 2) 