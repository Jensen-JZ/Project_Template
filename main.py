import sys

from munch import Munch

from config import load_cfg, print_cfg, save_cfg, setup_cfg, validate_cfg
from data.loader import get_selected_loader, get_test_loader, get_train_loader
from solver.solver import Solver


def main(args):
    if args.version:
        print("GeoCLIP Version: 1.0.0")
        print("Git Commit Hash:", args.hash)
        sys.exit()

    solver = Solver(args)
    if args.mode == "train":
        loaders = Munch(train=get_train_loader(**args), test=get_test_loader(**args))
        if args.selected_path:
            loaders.selected = get_selected_loader(**args)
        solver.train(loaders)
    elif args.mode == "sample":
        solver.sample()
    elif args.mode == "eval":
        solver.eval()
    else:
        assert False, f"Unimplemented Mode: {args.mode}"


if __name__ == "__main__":
    cfg = load_cfg()
    setup_cfg(cfg)
    validate_cfg(cfg)
    save_cfg(cfg)
    print_cfg(cfg)
    main(cfg)
