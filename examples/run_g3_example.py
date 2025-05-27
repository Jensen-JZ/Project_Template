# examples/run_g3_example.py
import torch
import os
from PIL import Image # For creating a dummy image
from transformers import CLIPTokenizer # For creating dummy text input if model's processor is problematic

# Attempt to import G3, handling potential import issues if run from outside project root
try:
    from models.G3 import G3
except ImportError:
    # This is a fallback for running the script directly from the examples folder
    # It assumes the project root is one level up and added to PYTHONPATH
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.G3 import G3

def create_dummy_image_tensor(vision_processor):
    """
    Creates a dummy PIL image and processes it with the vision_processor.
    If vision_processor is None or fails, returns a random tensor.
    """
    try:
        # Create a small, simple PIL image
        dummy_pil_image = Image.new('RGB', (224, 224), color = 'red')
        if vision_processor:
            # Process with the model's vision processor
            # The processor typically returns a dict with 'pixel_values'
            processed_image = vision_processor(images=dummy_pil_image, return_tensors='pt')['pixel_values']
            return processed_image.to('cpu') # Ensure it's on CPU
        else:
            print("Vision processor not available, creating random image tensor.")
            return torch.randn(1, 3, 224, 224, device='cpu')
    except Exception as e:
        print(f"Error creating dummy image with vision processor: {e}")
        print("Falling back to random image tensor.")
        return torch.randn(1, 3, 224, 224, device='cpu')

def create_dummy_text_input(text_processor):
    """
    Creates dummy text input using the model's text_processor (CLIPTokenizer).
    If text_processor is None or fails, returns a random tensor of typical shape.
    """
    dummy_text = ["a dummy caption"]
    try:
        if text_processor:
            # Process with the model's text processor
            # Expected output keys: 'input_ids', 'attention_mask'
            # We only need 'input_ids' and 'attention_mask' for the model's text_model part if separate
            # G3's text_model takes {'input_ids': ..., 'attention_mask': ...}
            inputs = text_processor(text=dummy_text, padding='max_length', truncation=True, return_tensors='pt', max_length=77)
            return {k: v.to('cpu') for k, v in inputs.items()} # Ensure on CPU
        else:
            print("Text processor not available, creating random text input tensors.")
            # Typical shape for CLIP input_ids and attention_mask
            input_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long, device='cpu') # 49408 is CLIP vocab size
            attention_mask = torch.ones((1, 77), dtype=torch.long, device='cpu')
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
    except Exception as e:
        print(f"Error creating dummy text input with text processor: {e}")
        print("Falling back to random text input tensors.")
        input_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long, device='cpu')
        attention_mask = torch.ones((1, 77), dtype=torch.long, device='cpu')
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

def run_g3_example():
    print("Initializing G3 model on CPU...")
    # 1. Initialize the G3 model on CPU
    device = torch.device('cpu')
    model = G3(device=device) # G3 class itself handles device placement internally for its submodules
    model.to(device) # Ensure the top-level model is on CPU

    # 2. Handle location_encoder.pth gracefully
    location_encoder_weights_path = 'location_encoder.pth'
    try:
        # The original G3 model loads state_dict for location_encoder like this:
        # location_encoder_dict = torch.load(location_encoder_weights_path, map_location=device)
        # model.location_encoder.load_state_dict(location_encoder_dict)
        # Since we don't have the file, we'll simulate the attempt or skip.
        # For this example, we assume the location_encoder is randomly initialized if file not found.
        if os.path.exists(location_encoder_weights_path):
            print(f"Found {location_encoder_weights_path}, attempting to load...")
            # In a real scenario, you would load. For this example, we'll just print.
            # model.location_encoder.load_state_dict(torch.load(location_encoder_weights_path, map_location=device))
            print(f"Mock loading of {location_encoder_weights_path}. In a real script, uncomment load_state_dict.")
        else:
            print(f"{location_encoder_weights_path} not found. Location encoder will use random initialization.")
    except FileNotFoundError:
        print(f"{location_encoder_weights_path} not found. Location encoder will use random initialization.")
    except Exception as e:
        print(f"An error occurred while trying to load location_encoder weights: {e}")
        print("Location encoder will use random initialization.")

    # Ensure model is in evaluation mode if not training
    model.eval()

    # 3. Prepare dummy inputs
    print("\nPreparing dummy inputs...")
    # Dummy Image Input
    # The G3 model initializes self.vision_processor = CLIPImageProcessor.from_pretrained(...)
    # This processor is callable.
    images_input = create_dummy_image_tensor(model.vision_processor)
    print(f"Dummy image tensor shape: {images_input.shape}")

    # Dummy Text Input
    # The G3 model initializes self.text_processor = CLIPTokenizer.from_pretrained(...)
    # This processor is callable.
    # The G3 forward method expects texts to be a dictionary like {'input_ids': ..., 'attention_mask': ...}
    texts_input = create_dummy_text_input(model.text_processor)
    print(f"Dummy text input_ids shape: {texts_input['input_ids'].shape}")
    print(f"Dummy text attention_mask shape: {texts_input['attention_mask'].shape}")


    # Dummy Longitude and Latitude
    longitude_input = torch.randn(1, device=device) # Batch size of 1
    latitude_input = torch.randn(1, device=device)   # Batch size of 1
    print(f"Dummy longitude tensor shape: {longitude_input.shape}")
    print(f"Dummy latitude tensor shape: {latitude_input.shape}")

    # 4. Perform a forward pass
    # The model's forward pass expects images, texts, longitude, latitude
    # And an optional return_loss argument.
    print("\nPerforming forward pass with return_loss=False...")
    with torch.no_grad(): # Important for inference to disable gradient calculations
        try:
            output = model(images_input, texts_input, longitude_input, latitude_input, return_loss=False)
            # 5. Print output information
            print("\nForward pass successful!")
            print("Output dictionary keys:", output.keys())
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f" - Output '{key}' shape: {value.shape}")
                elif value is None:
                    print(f" - Output '{key}' is None")
                else:
                    print(f" - Output '{key}' type: {type(value)}")
            
            # Example: Print a small part of one of the output tensors if it exists
            if 'image_embeds' in output and output['image_embeds'] is not None:
                print(f"   Sample of 'image_embeds': {output['image_embeds'][0, :5]}")
            if 'logits_per_texts_with_images' in output and output['logits_per_texts_with_images'] is not None:
                 print(f"   Sample of 'logits_per_texts_with_images': {output['logits_per_texts_with_images'][0, :5]}")


        except Exception as e:
            print(f"\nError during forward pass: {e}")
            print("This might be due to unexpected input shapes or issues within the model if not all parts were initialized.")

if __name__ == '__main__':
    run_g3_example()
    print("\nrun_g3_example.py finished.")
