# training/g3_trainer.py
import torch
import os
import numpy as np # Not strictly used in the refactored trainer, but was in original run_g3.py
import time # Not strictly used in the refactored trainer, but was in original run_g3.py
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings

# Attempt to import G3 and MP16Dataset, handling potential import issues
try:
    from models.G3 import G3
    from utils.g3_utils import MP16Dataset
except ImportError:
    import sys
    # Adjust path to project root if this script is in training/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.G3 import G3
    from utils.g3_utils import MP16Dataset

warnings.filterwarnings('ignore')

class G3Trainer:
    def __init__(self, batch_size=256, num_workers=16, learning_rate=3e-5, weight_decay=1e-6, scheduler_step_size=1, scheduler_gamma=0.87, checkpoint_dir='./checkpoints/'):
        """
        Initializes the G3Trainer.

        Args:
            batch_size (int): Batch size for the DataLoader.
            num_workers (int): Number of workers for the DataLoader.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            scheduler_step_size (int): Step size for the learning rate scheduler.
            scheduler_gamma (float): Gamma factor for the learning rate scheduler.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        print("Initializing G3Trainer...")

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.device = self.accelerator.device # Use device from Accelerator

        print(f"Device determined by Accelerator: {self.device}")

        # Model Initialization
        print("Initializing G3 model...")
        model = G3(device=self.device) # Pass self.device here

        # Loading location_encoder.pth
        location_encoder_weights_path = 'location_encoder.pth'
        try:
            if os.path.exists(location_encoder_weights_path):
                print(f"Loading location encoder weights from {location_encoder_weights_path}...")
                location_encoder_dict = torch.load(location_encoder_weights_path, map_location=self.device)
                model.location_encoder.load_state_dict(location_encoder_dict)
                print("Location encoder weights loaded successfully.")
            else:
                print(f"{location_encoder_weights_path} not found. Location encoder will use random initialization.")
        except Exception as e:
            print(f"Error loading location_encoder weights: {e}. Using random initialization.")

        # Dataset and DataLoader setup
        print("Setting up MP16Dataset and DataLoader...")
        dataset = MP16Dataset(vision_processor=model.vision_processor, text_processor=model.text_processor)
        # Note: shuffle=False was in original script for dataloader. Keeping it for consistency.
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=5 if num_workers > 0 else None)
        print(f"DataLoader created with batch size {batch_size} and {num_workers} workers.")

        # Optimizer and Scheduler setup
        print("Setting up Optimizer and Scheduler...")
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        print("Optimizer and Scheduler created.")

        # Prepare components with Accelerator
        print("Preparing components with Accelerator...")
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, self.optimizer, self.dataloader, self.scheduler
        )
        print("Accelerator preparation complete.")

        self.checkpoint_dir = checkpoint_dir
        if self.accelerator.is_local_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Checkpoint directory '{self.checkpoint_dir}' ensured.")
        
        print("G3Trainer initialization complete.")

    def _train_epoch(self, epoch_num):
        """
        Trains the model for one epoch.
        Args:
            epoch_num (int): The current epoch number.
        """
        self.model.train()
        t = tqdm(self.dataloader, disable=not self.accelerator.is_local_main_process, desc=f"Epoch {epoch_num+1}")
        
        total_loss = 0
        num_batches = 0

        for i, (images, texts, longitude, latitude) in enumerate(t):
            # The G3 model's internal text_processor needs to be accessed.
            # If self.model is the prepared model, unwrap it to access original attributes like text_processor.
            current_model_unwrapped = self.accelerator.unwrap_model(self.model)
            
            texts_processed = current_model_unwrapped.text_processor(text=texts, padding='max_length', truncation=True, return_tensors='pt', max_length=77)
            texts_processed = {k: v.to(self.device) for k, v in texts_processed.items()}

            images = images.to(self.device) # Ensure images are on device
            longitude = longitude.to(self.device).float()
            latitude = latitude.to(self.device).float()

            self.optimizer.zero_grad()
            
            output = self.model(images, texts_processed, longitude, latitude, return_loss=True)
            loss = output['loss']

            self.accelerator.backward(loss)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            if i % 1 == 0 and self.accelerator.is_local_main_process:
                t.set_postfix_str('loss {:.4f}, lr {:.6f}'.format(loss.item(), self.scheduler.get_last_lr()[0]), refresh=True)
        
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        if self.accelerator.is_local_main_process:
            print(f"Epoch {epoch_num+1} average loss: {avg_loss:.4f}")


    def train(self, num_epochs=10):
        """
        Main training loop.
        Args:
            num_epochs (int): Number of epochs to train.
        """
        if self.accelerator.is_local_main_process:
            print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            if self.accelerator.is_local_main_process:
                print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            self._train_epoch(epoch)

            if self.accelerator.is_local_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                checkpoint_path = os.path.join(self.checkpoint_dir, f'g3_epoch_{epoch+1}.pth')
                torch.save(unwrapped_model, checkpoint_path) # Original script saved the whole model
                print(f"Checkpoint saved to {checkpoint_path}")
        
        if self.accelerator.is_local_main_process:
            print("\nTraining complete.")
