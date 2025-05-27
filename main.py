# main.py - Generic PyTorch Template Entry Point
# -----------------------------------------------------------------------------
# This script serves as the main entry point for the generic PyTorch template
# from which this project was partially derived. It is designed to work with
# the configuration system in `config.py` and the generic training/evaluation
# loops provided in `solver/solver.py`.
#
# To use this generic template framework:
# 1. Define your model architecture in `models/build.py`.
# 2. Define your loss function(s) in `solver/loss.py`.
# 3. Configure your experiment via command-line arguments (defined in `config.py`)
#    or by providing a JSON configuration file.
#
# **Important Note for G3 Model Operations**:
# The core functionalities and specific workflows for the G3 geolocalization
# model (e.g., G3 training, FAISS indexing, LLM-based geolocation predictions)
# are NOT primarily run through this `main.py` script.
#
# Instead, please use the dedicated launcher scripts located in the `scripts/`
# directory. For more details, refer to `scripts/README.md` or the main
# project `README.md`.
# -----------------------------------------------------------------------------
from munch import Munch

from config import setup_cfg, validate_cfg, load_cfg, save_cfg, print_cfg
from data.loader import get_train_loader, get_test_loader # Removed get_selected_loader
from solver.solver import Solver


def main(args):
    solver = Solver(args)
    if args.mode == 'train':
        loaders = Munch(train=get_train_loader(**args), test=get_test_loader(**args))
        # Removed selected_path and get_selected_loader logic
        solver.train(loaders)
    elif args.mode == 'sample':
        # solver.sample() now expects a loaders argument.
        # Typically, for sampling, one might use the test_loader or a specific inference loader.
        # For simplicity, we'll pass a loader with the test set.
        # User might need to customize this based on their sampling needs.
        loaders = Munch(test=get_test_loader(**args))
        solver.sample(loaders)
    elif args.mode == 'eval':
        # solver.evaluate() now expects a loaders argument.
        loaders = Munch(test=get_test_loader(**args))
        solver.evaluate(loaders)
    else:
        assert False, f"Unimplemented mode: {args.mode}"


if __name__ == '__main__':
    cfg = load_cfg()
    setup_cfg(cfg)
    validate_cfg(cfg)
    save_cfg(cfg)
    print_cfg(cfg)
    main(cfg)
