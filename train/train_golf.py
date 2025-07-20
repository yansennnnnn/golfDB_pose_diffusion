import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from finetune.schemas import Args
from finetune.trainer_golf import GolfFinetuneTrainer

def main():
    """
    Main entry point for training the structure-aware golf swing video generation model.
    """
    # Note: For a real implementation, you would extend Args to include
    # specific paths for the golf dataset, keyframes, etc.
    args = Args.parse_args()
    
    # Instantiate and run the specialized trainer
    trainer = GolfFinetuneTrainer(args)
    trainer.fit()

if __name__ == "__main__":
    main() 