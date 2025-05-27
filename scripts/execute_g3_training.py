# scripts/execute_g3_training.py
import argparse # Included for future flexibility, though not used in this version
import os
import sys

# Adjust path to project root to find the 'training' module
# This assumes the script is in 'scripts/' and 'training/' is at the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from training.g3_trainer import G3Trainer
except ImportError as e:
    print(f"Error importing G3Trainer: {e}")
    print("Ensure that training/g3_trainer.py exists and the PYTHONPATH is set correctly.")
    print("If running from the 'scripts' directory, the project root should be one level up.")
    sys.exit(1)

def main():
    # Future: Parse arguments here if needed
    # parser = argparse.ArgumentParser(description="Execute G3 Model Training")
    # parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    # parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    # ... other arguments from G3Trainer.__init__
    # args = parser.parse_args()

    print("Creating G3Trainer instance...")
    # For now, use default parameters as in the original script for num_epochs
    # and G3Trainer's defaults for other params.
    trainer = G3Trainer(
        # batch_size=args.batch_size, # Example for future use
        # num_workers=16, # Default in G3Trainer
        # learning_rate=3e-5, # Default in G3Trainer
        # checkpoint_dir='./checkpoints/' # Default in G3Trainer
    )
    
    print("Starting training process...")
    trainer.train(num_epochs=10) # num_epochs was 10 in the original script's loop
    print("Training process finished.")

if __name__ == '__main__':
    main()
