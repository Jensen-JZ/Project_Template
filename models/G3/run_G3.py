import torch
import os
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import MP16Dataset
from utils.G3 import G3
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings

warnings.filterwarnings('ignore')

def train_1epoch(dataloader, eval_dataloader, earlystopper, model, vision_processor, text_processor, optimizer, scheduler, device, accelerator=None):
    model.train()
    t = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    for i, (images, texts, longitude, latitude) in enumerate(t):
        texts = text_processor(text=texts, padding='max_length', truncation=True, return_tensors='pt', max_length=77)
        images = images.to(device)
        texts = texts.to(device)
        longitude = longitude.to(device).float()
        latitude = latitude.to(device).float()
        optimizer.zero_grad()

        output = model(images, texts, longitude, latitude, return_loss=True)
        loss = output['loss']

        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        if i % 1 == 0:
            t.set_description('step {}, loss {}, lr {}'.format(i, loss.item(), scheduler.get_last_lr()[0]))
    scheduler.step()


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # fine-tune
    # device = "cuda" if torch.cuda.is_available() else "cpu" # Original line
    device = 'cpu' # Forcing CPU for testing without GPU
    print(f"G3 Script: Forcing device to {device}")
    model = G3(device) # Initialize model on CPU first
    # location_encoder_dict = torch.load('location_encoder.pth') # from geoclip
    # For testing, create a dummy state_dict if file is empty or invalid
    try:
        location_encoder_dict = torch.load('location_encoder.pth', map_location=device)
        if not location_encoder_dict: # If file is empty, torch.load might return None or empty dict
            print("G3 Script: location_encoder.pth is empty or invalid. Using dummy state_dict.")
            # Create a dummy state_dict based on G3's location_encoder structure if possible
            # This is a placeholder, actual structure might be complex.
            # Assuming it's a simple nn.Linear for placeholder purposes.
            # model.location_encoder is likely more complex.
            # This part might still cause issues if the dummy doesn't match expectations.
            # For now, we'll try loading and if it fails or is empty, we just print a warning.
            # And let it proceed (it might fail later in load_state_dict).
            pass # Fall through to load_state_dict, which might fail if dict is truly empty/bad
    except Exception as e:
        print(f"G3 Script: Failed to load or process location_encoder.pth: {e}. Using dummy state_dict concept.")
        # As a last resort, if torch.load fails (e.g. empty file), create an empty dict
        # This will likely cause load_state_dict to fail, but it helps pinpoint the issue.
        location_encoder_dict = {} 
        # It's better to let load_state_dict fail clearly than to hide it with a bad dummy.

    # Only load if location_encoder_dict is not empty and seems valid
    if location_encoder_dict:
        try:
            model.location_encoder.load_state_dict(location_encoder_dict)
        except Exception as e:
            print(f"G3 Script: Error loading state_dict for location_encoder: {e}. Model may not work correctly.")
    else:
        print("G3 Script: Skipping load_state_dict for location_encoder due to empty/invalid location_encoder.pth.")


    dataset = MP16Dataset(vision_processor = model.vision_processor, text_processor = model.text_processor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)


    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            params.append(param)

    optimizer = torch.optim.AdamW([param for name,param in model.named_parameters() if param.requires_grad], lr=3e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.87)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    # model = model.to(device) # Ensure model is on the correct device after prepare

    eval_dataloader = None
    earlystopper = None
    
    # Ensure checkpoint directory exists
    os.makedirs('checkpoints', exist_ok=True) # Added line

    for epoch in range(10):
        train_1epoch(dataloader, eval_dataloader, earlystopper, model, model.vision_processor, model.text_processor, optimizer, scheduler, device, accelerator)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model, 'checkpoints/g3_{}_.pth'.format(epoch))

if __name__ == '__main__':
    main()
