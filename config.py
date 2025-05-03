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

from utils.file import list_sub_folders, prepare_dirs, save_json
from utils.misc import get_commit_hash, get_datetime, start_tensorboard, str2bool


def setup_cfg(args):
    """
    Prepare and initialize runtime configurations before training or evaluation,
    including random seeds, device settings, experiment ID resolution, directory creation, and TensorBoard launching.

    Args:
        args (Namespace or Munch): Parsed command-line arguments.

    Returns:
        None: Modifies `args` in-place with additional derived fields.
    """

    cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Suppress TensorFlow INFO and WARNING logs; Only show ERROR messages.
    # Example ==> {'0': ALL, '1': INFO, '2': WARNING, '3': ERROR}
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if args.mode == "train" and torch.cuda.device_count() > 1:
        print(f"We will use {torch.cuda.device_count()} GPUs for training.")
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.mode == "train":
        if args.exp_id is None:
            args.exp_id = get_datetime()
    else:
        if args.exp_id is None:
            args.exp_id = input(
                "Please provide an experiment ID for evaluation or sampling: "
            )
        if not os.path.exists(os.path.join(args.exp_dir, args.exp_id)):
            all_existed_ids = os.listdir(args.exp_dir)
            for existed_id in all_existed_ids:
                #
                if existed_id.startswith(args.exp_id + "_"):
                    args.exp_id = existed_id
                    print(
                        f"Warning: Experiment ID not found. Using existing experiment folder: {args.exp_id}"
                    )
                    break

    if args.debug:
        print("Warning: Debug mode is enabled. Some settings may be overridden.")
        args.exp_id = "debug"
        args.sample_every = 10
        args.eval_every = 20
        args.save_every = 20
        args.end_iter = args.start_iter + 60
    if os.name == "nt" and args.num_workers != 0:
        # On Windows, DataLoader with num_workers > 0 may cause issues due to lack of fork support.
        # Setting num_workers to 0 to avoid this issue.
        print("Warning: Windows OS detected. Setting num_workers to 0.")
        args.num_workers = 0

    args.log_dir = os.path.join(args.exp_dir, args.exp_id, "logs")
    args.sample_dir = os.path.join(args.exp_dir, args.exp_id, "samples")
    args.model_dir = os.path.join(args.exp_dir, args.exp_id, "models")
    args.eval_dir = os.path.join(args.exp_dir, args.exp_id, "eval")
    prepare_dirs([args.log_dir, args.sample_dir, args.model_dir, args.eval_dir])
    print(
        f"Experiment outputs will be saved to: {os.path.join(args.exp_dir, args.exp_id)}"
    )
    print(f"🔧 Experiment ID: {args.exp_id}")
    print(f"🚀 Mode: {args.mode.upper()}, Device: {args.device}")
    args.record_file = os.path.join(args.exp_dir, args.exp_id, "record.txt")
    args.loss_file = os.path.join(args.exp_dir, args.exp_id, "losses.csv")

    if os.path.exists(f"./scripts/{args.exp_id}.sh"):
        shutil.copyfile(
            f"./scripts/{args.exp_id}.sh",
            os.path.join(args.exp_dir, args.exp_id, f"{args.exp_id}.sh"),
        )

    if args.mode == "train" and args.start_tensorboard:
        start_tensorboard(os.path.join(args.exp_dir, args.exp_id), "logs")

    # Automatically detect domains names from subfolders in the training dataset.
    # Each subfolder (e.g., 'sunny', 'cloudy', 'night') is treated as one domain.
    args.domains = list_sub_folders(args.train_path, full_path=False)
    args.num_domains = len(args.domains)


def validate_cfg(args):
    """
    Validates key configurations constraints before running the experiment.
    This includes checking the consistency of evaluation and saving frequency,
    and ensuring the declared number of domains matches the dataset structure.

    Args:
        args (Namespace or Munch): Parsed command-line arguments.

    Returns:
        None: Raises AssertionError if any constraint is violated.
    """

    # Ensure that evaluation only occurs after a ckpt has been saved.
    assert (
        args.eval_every % args.save_every == 0
    ), "eval_every must be a multiple of save_every to ensure ckpt exists before evaluation."
    # Ensure the number of declared domains matches the number of folders in the test set.
    assert args.num_domains == len(list_sub_folders(args.test_path, full_path=False))


def load_cfg():
    """
    Loads experiment configurations from a JSON file or command-line arguments.

    If a JSON file is passed as the first command-line argument, loads configurations from that file.
    Otherwise, falls back to parsing command-line arguments via argparse.
    Also verifies Git commit has consistency to ensure reproducibility.

    Returns:
        Munch: A configuration object with all parsed and derived settings.
    """

    # There are two ways to load the configuration:
    # 1. From a JSON file passed as the first command-line argument.
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1], "r") as f:
            cfg = json.load(f)
            cfg = Munch(cfg)
            if len(sys.argv) >= 3:
                cfg.exp_id = sys.argv[2]
            else:
                print(
                    "Warning: No experiment ID provided. Using existing experiment folder."
                )
            if not cfg.about:
                cfg.about = f"Copied from: {sys.argv[1]}"
    # 2. From command-line arguments using argparse.
    else:
        cfg = parse_args()
        cfg = Munch(cfg.__dict__)
        if not cfg.hash:
            cfg.hash = get_commit_hash()
    current_hash = get_commit_hash()
    if current_hash != cfg.hash:
        print(f"Warning: Unmatched git commit hash: `{current_hash}` & `{cfg.hash}`.")

    return cfg


def save_cfg(cfg):
    """
    Saves the current experiment configuration to a JSON file.

    The configuration is saved under `cfg.exp_dir/cfg.exp_id`, with the filename,
    based on the current mode (e.g., 'train.json', 'eval.json'). If training is resumed
    from a non-zero iteration, the filename is set to 'resume_<start_iter>.json'.

    Args:
        cfg (Munch): Experiment configuration object containing all settings and parameters.

    Returns:
        None: Saves the configuration to a JSON file.
    """

    exp_path = os.path.join(cfg.exp_dir, cfg.exp_id)
    os.makedirs(exp_path, exist_ok=True)
    filename = cfg.mode
    if cfg.mode == "train" and cfg.start_iter != 0:
        filename = f"resume_{cfg.start_iter}"
    save_json(exp_path, cfg, filename)


def print_cfg(cfg):
    """
    Prints the current experiment configuration in a readable JSON format.

    Useful for logging or debugging to verify the full set of active parameters and settings.

    Args:
        cfg (dict or Munch): The configuration object to display.

    Returns:
        None: Prints the configuration to the console.
    """

    print(json.dumps(cfg, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()

    # Meta / Global Settings
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file (e.g., JSON or YAML)to override default arguments.",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version information and exit."
    )

    # Experiment Information
    parser.add_argument("--about", type=str, default="")
    parser.add_argument("--hash", type=str, help="Git commit hash for this experiment.")
    parser.add_argument(
        "--exp_id",
        type=str,
        help="Unique identifier for this experiment. Used to name output dirs, logs, ckpts, etc.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="expr",
        help="Root dir for saving all experiment outputs such as logs, ckpts, and samples.",
    )

    # Runtime Mode
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval", "sample"]
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Dataset & Data Settings
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument(
        "--compare_path",
        type=str,
        required=True,
        help="Path to reference images for computing evaluation metrics like LPIPS or FID.",
    )
    parser.add_argument(
        "--selected_path",
        type=str,
        help="Optional path to a subset of images selected for sampling or evaluation.",
    )
    parser.add_argument(
        "--num_domains",
        type=int,
        help="Number of domains or categories involved in training. Used for multi-domain generation or classification tasks.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        help="List of domain names involved in training or evaluaton. Used for conditional generation or domain-specific sampling.",
    )

    # Model & Training Configuration
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=16,
        help="Dimensionality of the latent vertor z used for sampling or generation.",
    )
    parser.add_argument(
        "--style_dim",
        type=int,
        default=64,
        help="Dimensionality of the style vertor used to control image generation.",
    )
    parser.add_argument(
        "--init_weights", type=str, default="he", choices=["he", "default"]
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--end_iter", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--d_lr", type=float, default=1e-4, help="Learning rate for the discriminator."
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.0,
        help="Beta1 coefficient for Adam optimizer (momentum term). Controls update stability.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.99,
        help="Same as Beta1. Controls how fast the optimizer forgets past gradients.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) applied to optimizer.",
    )
    parser.add_argument(
        "--ema_beta",
        type=float,
        default=0.999,
        help="EMA decay rate for model parameters. Higher == slower updates.",
    )
    parser.add_argument(
        "--lambda_adv",
        type=float,
        default=10,
        help="Weight for the adversarial loss term. Controls how much the GAN loss contributes to the total loss.",
    )
    parser.add_argument(
        "--lambda_r1",
        type=float,
        default=1,
        help="Weight for the R1 regularization term.",
    )
    parser.add_argument(
        "--which_lpips",
        type=str,
        default="alex",
        choices=["vgg", "alex"],
        help="LPIPS model to use for perceptual loss.",
    )

    # Evaluation & Sampling Configuration
    parser.add_argument(
        "--sample_id",
        type=str,
        help="Optional identifier for sample output. Used to name result files or folders during sample mode.",
    )
    parser.add_argument(
        "--sample_non_ema",
        type=str2bool,
        default=True,
        help="Whether to also sample non-EMA (Exponential Moving Average) model during evaluation or sample mode.",
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        help="Iteration number to load model for evaluation. If not set, defaults to latest checkpoint.",
    )
    parser.add_argument(
        "--eval_use_ema",
        type=str2bool,
        default=True,
        help="Whether to use EMA (Exponential Moving Average) version of the model during evaluation.",
    )
    parser.add_argument(
        "--keep_all_eval_samples",
        type=str2bool,
        default=False,
        help="If True, keep all evalutation samples. If False, may overwrite or discard intermediate outputs.",
    )
    parser.add_argument(
        "--keep_best_eval_samples",
        type=str2bool,
        default=True,
        help="Whether to save only the best-performing evaluation samples based on selected metrics.",
    )
    parser.add_argument(
        "--eval_repeat_num",
        type=int,
        default=1,
        help="Number of times to repeat the evaluation process. This is useful for obtaining more stable metrics by averaging results over multiple runs.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument(
        "--eval_cache",
        type=str2bool,
        default=True,
        help="Whether to cache evaluation-related files (e.g., features or intermediate results) to speed up repeated runs.",
    )
    parser.add_argument(
        "--eval_max_num",
        type=int,
        help="Maximum number of samples to evaluate. Useful for partial evaluation or debugging.",
    )

    # Logging & Utilities
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Interval (in training steps) at which to print logs such as loss, metrics, or progress.",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=1000,
        help="Interval (in training steps) at which to generate and save sample outputs from the model.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="Interval (in training steps) at which to save model ckpts.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5000,
        help="Interval (in training steps) at which to run evaluation on the test set or validation set.",
    )
    parser.add_argument(
        "--use_tensorboard",
        type=str2bool,
        default=True,
        help="Whether to use TensorBoard for logging.",
    )
    parser.add_argument(
        "--start_tensorboard",
        type=str2bool,
        default=False,
        help="Whether to launch TensorBoard logging automatically at training start.",
    )
    parser.add_argument(
        "--save_loss",
        type=str2bool,
        default=True,
        help="Whether to save loss values during training for later analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Set to 0 for random seed.",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        type=str2bool,
        default=True,
        help="Enables cudnn.benchmark for potential speedup if input sizes are constant.",
    )
    parser.add_argument(
        "--keep_all_models",
        type=str2bool,
        default=True,
        help="Whether to keep all model ckpts during training. If False, only the last or best may be kept.",
    )
    parser.add_argument(
        "--pretrained_models",
        type=str,
        nargs="+",
        default=[],
        help="List of pretrained model names or paths to load before training or evaluation.",
    )

    return parser.parse_args()
