from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder # Use official ImageFolder
from .g3_datasets import MP16Dataset # Added import

from data.dataset import DefaultDataset
# Removed: from utils.file import list_all_images - not needed after removing get_selected_loader
# Removed: import numpy as np - not needed after removing _make_balanced_sampler
# Removed: from torch.utils.data.sampler import WeightedRandomSampler - not needed


def get_train_loader(args, **unused_kwargs):
    if args.model_name == 'g3':
        print("Using MP16Dataset for G3 model training.")
        # Expected args for G3 MP16Dataset (to be added to config.py later)
        # For robustness during development, provide fallbacks or ensure args exist.
        # For this subtask, we'll assume they will be added to args by config.py.
        dataset = MP16Dataset(
            root_path=getattr(args, 'g3_dataroot', './data/'), # Default if not in args
            text_data_path=getattr(args, 'g3_mp16_csv_name', 'MP16_Pro_places365.csv'),
            image_data_path=getattr(args, 'g3_mp16_tar_name', 'mp-16-images.tar'),
            member_info_path=getattr(args, 'g3_mp16_tar_index_name', 'tar_index.pkl'),
            clip_model_name=getattr(args, 'g3_clip_model_name', "openai/clip-vit-large-patch14")
            # vision_processor and text_processor are left as None to use dataset defaults
        )
        return data.DataLoader(dataset=dataset,
                               batch_size=args.batch_size, # Use args.batch_size
                               shuffle=True,
                               num_workers=args.num_workers, # Use args.num_workers
                               pin_memory=True,
                               drop_last=True)
    else:
        print(f"Using ImageFolder dataset from {args.train_path}")
        transform = transforms.Compose([
            transforms.Resize([args.input_shape[0], args.input_shape[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = ImageFolder(root=args.train_path, transform=transform)
        return data.DataLoader(dataset=dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=True)


def get_test_loader(args, **unused_kwargs):
    if args.model_name == 'g3':
        print("Using MP16Dataset for G3 model testing.")
        dataset = MP16Dataset(
            root_path=getattr(args, 'g3_dataroot', './data/'),
            text_data_path=getattr(args, 'g3_mp16_csv_name', 'MP16_Pro_places365.csv'), # Or a specific test CSV if different
            image_data_path=getattr(args, 'g3_mp16_tar_name', 'mp-16-images.tar'), # Or a specific test TAR if different
            member_info_path=getattr(args, 'g3_mp16_tar_index_name', 'tar_index.pkl'), # Or a specific test PKL
            clip_model_name=getattr(args, 'g3_clip_model_name', "openai/clip-vit-large-patch14")
        )
        return data.DataLoader(dataset=dataset,
                               batch_size=args.test_batch_size, # Use args.test_batch_size
                               shuffle=False,
                               num_workers=args.num_workers,
                               pin_memory=True)
    else:
        print(f"Using ImageFolder dataset from {args.test_path}")
        transform = transforms.Compose([
            transforms.Resize([args.input_shape[0], args.input_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = ImageFolder(root=args.test_path, transform=transform) # Use args.test_path
        return data.DataLoader(dataset=dataset,
                               batch_size=args.test_batch_size,
                               shuffle=False,
                               num_workers=args.num_workers,
                               pin_memory=True)


def get_eval_loader(args, **unused_kwargs):
    # Path should be an image folder without sub folders.
    # Generic transform
    transform = transforms.Compose([
        transforms.Resize([args.input_shape[0], args.input_shape[1]]), # Use args.input_shape
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Generic normalization
    ])
    # Ensure eval_path exists on args, or provide a default/handle error
    eval_path = getattr(args, 'eval_path', getattr(args, 'test_path', './data/default_eval')) # Fallback for eval_path
    print(f"Using DefaultDataset for evaluation from {eval_path}")
    dataset = DefaultDataset(root=eval_path, transform=transform) # DefaultDataset for flat folder structure

    return data.DataLoader(dataset=dataset,
                           batch_size=args.test_batch_size, # Use args.test_batch_size
                           shuffle=False,
                           num_workers=args.num_workers, # Use args.num_workers
                           pin_memory=True,
                           drop_last=False)

# Removed get_selected_loader as selected_path was removed from config
# Removed _make_balanced_sampler as it's not universally needed
