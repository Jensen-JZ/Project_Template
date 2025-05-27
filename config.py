import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
from munch import Munch
from torch.backends import cudnn

from utils.file import prepare_dirs, list_sub_folders
from utils.file import save_json
from utils.misc import get_datetime, str2bool, get_commit_hash, start_tensorboard


def setup_cfg(args):
    # Use getattr to provide defaults for attributes that might be missing if loaded from JSON
    # These defaults match what's in parse_args()
    cudnn.benchmark = getattr(args, 'cudnn_benchmark', True)
    _seed = getattr(args, 'seed', 0)
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    _mode = getattr(args, 'mode', 'train') # Default mode to 'train' if not specified

    if _mode == 'train' and torch.cuda.device_count() > 1:
        print(f"We will train on {torch.cuda.device_count()} GPUs.")
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if _mode == 'train':
        if not hasattr(args, 'exp_id') or args.exp_id is None:
            args.exp_id = get_datetime()
            # Tip: you can construct the exp_id automatically here by use the args.
    else: # eval or sample mode
        if not hasattr(args, 'exp_id') or args.exp_id is None:
            args.exp_id = input("Please input exp_id: ")
        _exp_dir = getattr(args, 'exp_dir', 'expr') # Default exp_dir
        if not os.path.exists(os.path.join(_exp_dir, args.exp_id)):
            all_existed_ids = os.listdir(_exp_dir)
            for existed_id in all_existed_ids:
                if existed_id.startswith(args.exp_id + "-"):
                    args.exp_id = existed_id
                    print(f"Warning: exp_id is reset to {existed_id}.")
                    break
    
    _exp_dir = getattr(args, 'exp_dir', 'expr') # Ensure exp_dir has a default

    if getattr(args, 'debug', False): # Default debug to False
        print("Warning: running in debug mode, some settings will be override.")
        args.exp_id = "debug"
        args.sample_every = getattr(args, 'sample_every', 10)
        args.eval_every = getattr(args, 'eval_every', 20)
        args.save_every = getattr(args, 'save_every', 20)
        _start_iter = getattr(args, 'start_iter', 0)
        args.end_iter = _start_iter + 60
    
    _num_workers = getattr(args, 'num_workers', 4) # Default num_workers
    if os.name == 'nt' and _num_workers != 0:
        print("Warning: reset num_workers = 0, because running on a Windows system.")
        args.num_workers = 0

    args.log_dir = os.path.join(_exp_dir, args.exp_id, "logs")
    args.sample_dir = os.path.join(_exp_dir, args.exp_id, "samples")
    args.model_dir = os.path.join(_exp_dir, args.exp_id, "models")
    args.eval_dir = os.path.join(_exp_dir, args.exp_id, "eval")
    prepare_dirs([args.log_dir, args.sample_dir, args.model_dir, args.eval_dir])
    args.record_file = os.path.join(_exp_dir, args.exp_id, "records.txt")
    args.loss_file = os.path.join(_exp_dir, args.exp_id, "losses.csv")

    if _mode == 'train' and getattr(args, 'start_tensorboard', False): # Default start_tensorboard to False
        start_tensorboard(os.path.join(_exp_dir, args.exp_id), 'logs')


def validate_cfg(args):
    # Use getattr to provide defaults for attributes that might be missing
    _eval_every = getattr(args, 'eval_every', 5000)
    _save_every = getattr(args, 'save_every', 5000)
    assert _eval_every % _save_every == 0


def load_cfg():
    # There are two ways to load config, use a json file or command line arguments.
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1], 'r') as f:
            cfg = json.load(f)
            cfg = Munch(cfg)
            if len(sys.argv) >= 3:
                cfg.exp_id = sys.argv[2]
            else:
                print("Warning: using existing experiment dir.")
            if not cfg.about:
                cfg.about = f"Copied from: {sys.argv[1]}"
            if not hasattr(cfg, 'hash') or not cfg.hash: # Check if hash is missing or empty
                print("Warning: 'hash' not found in JSON config, adding current git commit hash.")
                cfg.hash = get_commit_hash()
            if not hasattr(cfg, 'exp_dir') or not cfg.exp_dir: # Check if exp_dir is missing or empty
                print("Warning: 'exp_dir' not found in JSON config, setting to default 'expr'.")
                cfg.exp_dir = 'expr'
    else:
        cfg = parse_args()
        cfg = Munch(cfg.__dict__)
        if not cfg.hash:
            cfg.hash = get_commit_hash()
    current_hash = get_commit_hash()
    if current_hash != cfg.hash:
        print(f"Warning: unmatched git commit hash: `{current_hash}` & `{cfg.hash}`.")
    return cfg


def save_cfg(cfg):
    exp_path = os.path.join(cfg.exp_dir, cfg.exp_id)
    os.makedirs(exp_path, exist_ok=True)
    filename = cfg.mode
    if cfg.mode == 'train' and cfg.start_iter != 0:
        filename = f"resume_{cfg.start_iter}"
    save_json(exp_path, cfg, filename)


def print_cfg(cfg):
    print(json.dumps(cfg, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()

    # About this experiment.
    parser.add_argument('--about', type=str, default="")
    parser.add_argument('--hash', type=str, required=False, help="Git commit hash for this experiment.")
    parser.add_argument('--exp_id', type=str, help='Folder name and id for this experiment.')
    parser.add_argument('--exp_dir', type=str, default='expr')

    # Meta arguments.
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'sample'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model related arguments.
    parser.add_argument('--model_name', type=str, default='default_model', choices=['default_model', 'g3'], help='Name of the model architecture to use.')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[256, 256], help='Shape of the input data, e.g., H W for images')
    parser.add_argument('--init_weights', type=str, default='he', choices=['he', 'default'])

    # Dataset related arguments.
    parser.add_argument('--dataset', type=str, required=True, help="Identifier or path to the dataset.")
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)

    # Training related arguments
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--end_iter', type=int, default=200000)

    # Evaluation related arguments
    parser.add_argument('--test_batch_size', type=int, default=32)

    # Optimizing related arguments.
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Step related arguments.
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--visualize_every', type=int, default=1000, help="Frequency of visualizing samples during training.")
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=5000)

    # Log related arguments.
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--start_tensorboard', type=str2bool, default=False)
    parser.add_argument('--save_loss', type=str2bool, default=True)

    # Others
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True)
    parser.add_argument('--keep_all_models', type=str2bool, default=True)
    parser.add_argument('--pretrained_models', type=str, nargs='+', default=[],
                        help='The name list of the pretrained models that you used.')

    return parser.parse_args()
