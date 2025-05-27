import torch.nn as nn
import torch # Added import
from munch import Munch
from .g3_architecture import G3 # Added import

def build_model(args):
    if args.model_name == 'g3':
        g3_model = G3(device_str=args.device) # Assuming G3 constructor expects device_str
        if hasattr(args, 'g3_location_encoder_path') and args.g3_location_encoder_path:
            print(f"Loading G3 location encoder weights from: {args.g3_location_encoder_path}")
            # Ensure map_location is used if the model might be loaded on a different device than saved
            location_encoder_dict = torch.load(args.g3_location_encoder_path, map_location=torch.device(args.device))
            g3_model.location_encoder.load_state_dict(location_encoder_dict)
            print("G3 location encoder weights loaded successfully.")
        else:
            print("Warning: `g3_location_encoder_path` not provided or empty. G3 location encoder weights not loaded.")
        
        nets = Munch(model=g3_model)
        nets_ema = Munch(model_ema=None) # Or a deepcopy for EMA
        return nets, nets_ema
    else:
        # TODO: User should define their model(s) here and return them.
        # This is a placeholder. Replace with your actual model.
        # Example:
        # model = nn.Sequential(
        #     nn.Linear(args.input_shape[0] * args.input_shape[1], 512), # Assuming input_shape is H, W
        #     nn.ReLU(),
        #     nn.Linear(512, 10) # Example output size
        # )
        # nets = Munch(model=model)
        # nets_ema = Munch(model_ema=None) # Or your EMA model if you use one
        # return nets, nets_ema
        print("Please define your model loading logic in models/build.py for other models.") # Modified print statement
        # Returning placeholder simple models for now to avoid errors in solver.py
        # These will need to be adjusted or properly handled by the user.
        model = nn.Linear(10,10) # Placeholder
        nets = Munch(model=model)
        nets_ema = Munch(model_ema=model) # Placeholder for EMA
        return nets, nets_ema
