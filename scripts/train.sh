#!/bin/bash
# A basic training script example.
# Users should customize paths and arguments according to their dataset and needs.

# Extracting a base name for the experiment ID from the script's filename.
filename=$(basename "$0")
exp_id="${filename%.*}_$(date +%Y%m%d_%H%M%S)" # Add timestamp for uniqueness

# Ensure you have a Conda environment or virtual environment activated
# with all necessary dependencies installed (e.g., from requirements.txt).

# Example:
# conda activate your_env_name
# pip install -r requirements.txt

echo "Starting training with Experiment ID: $exp_id"

# Customize these paths for your dataset
TRAIN_DATA_PATH="./data/placeholder_dataset/train"
TEST_DATA_PATH="./data/placeholder_dataset/test"
DATASET_NAME="MyDataset" # Or a generic name if dataset arg isn't used for logic

# Create placeholder directories and a dummy file if they don't exist to prevent immediate errors
# when ImageFolder or DefaultDataset try to load data.
# Users should replace these with their actual data.
mkdir -p $TRAIN_DATA_PATH
mkdir -p $TEST_DATA_PATH
touch $TRAIN_DATA_PATH/.gitkeep # Ensures ImageFolder doesn't complain if dir is empty
touch $TEST_DATA_PATH/.gitkeep  # Ensures ImageFolder doesn't complain if dir is empty
# For DefaultDataset, you might need at least one actual image file.
# Example:
# convert -size 10x10 xc:white $TRAIN_DATA_PATH/placeholder_img.png
# convert -size 10x10 xc:white $TEST_DATA_PATH/placeholder_img.png


# Common arguments for training
# Adjust batch_size, input_shape, and other parameters as needed.
# --input_shape is H W (e.g., 256 256 for 256x256 images)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} # Default to GPU 0 if not set

python main.py \
  --exp_id "$exp_id" \
  --about "Generic training example using train.sh" \
  --mode "train" \
  --start_iter 0 \
  --end_iter 100000 \
  --use_tensorboard true \
  --save_loss true \
  --log_every 50 \
  --eval_every 1000 \
  --save_every 5000 \
  --visualize_every 1000 \
  --dataset "$DATASET_NAME" \
  --train_path "$TRAIN_DATA_PATH" \
  --test_path "$TEST_DATA_PATH" \
  --batch_size 8 \
  --input_shape 256 256 \
  --lr 1e-4 \
  # Add other relevant arguments from your config.py as needed
  # e.g. --num_workers 4, --device 'cuda' / 'cpu'

echo "Training script finished for Experiment ID: $exp_id"
# Example: to run this script, navigate to the project root and execute:
# bash scripts/train.sh
