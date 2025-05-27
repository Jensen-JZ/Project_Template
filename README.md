# PyTorch Deep Learning Template

## Introduction

This project provides a generic and streamlined PyTorch template designed to accelerate the development of various deep learning models. It has been refactored from a previous GAN-specific template to offer a clean, well-organized starting point for a wide range of tasks, such as classification, regression, etc.

The template includes:
- A clear project structure.
- Configuration management via `config.py` and command-line arguments.
- Basic training, evaluation, and sampling loops in `solver/solver.py`.
- Placeholder for model definition in `models/build.py`.
- Placeholder for loss computation in `solver/loss.py`.
- Utility functions for logging (TensorBoard), checkpointing, and file operations.
- An example training script in `scripts/train.sh`.

## Project Structure

The project is organized into the following main directories and files:

-   **`README.md`**: This file, providing an overview and guide.
-   **`requirements.txt`**: Lists project dependencies.
-   **`main.py`**: The main entry point for running training, evaluation, or sampling.
-   **`config.py`**: Defines and manages all configuration options and arguments.
-   **`data/`**: Contains data loading and preprocessing utilities.
    -   `dataset.py`: Includes `DefaultDataset` for loading images from a flat folder.
    -   `loader.py`: Provides `get_train_loader`, `get_test_loader`, `get_eval_loader` using generic `ImageFolder` or `DefaultDataset`.
    -   `fetcher.py`: A wrapper for data loaders to provide an infinite iterator and move data to the device.
-   **`expr/`**: Default directory for storing experiment outputs (logs, models, samples).
-   **`models/`**: For model architecture definitions.
    -   `build.py`: **Placeholder.** Users must define their model(s) here. It should return a `Munch` object containing the model (e.g., `nets.model`).
    -   `README.md`: Briefly explains the purpose of this directory.
-   **`solver/`**: Handles the training, evaluation, and sampling logic.
    -   `solver.py`: Contains the main `Solver` class with generic training/evaluation loops.
    -   `loss.py`: **Placeholder.** Contains a `compute_loss` function where users must define their loss calculation.
    -   `utils.py`: Utilities for the solver, like weight initialization.
    -   `misc.py`: Placeholder for miscellaneous solver utilities.
-   **`metrics/`**: **Placeholder.** For user-defined metric calculations (e.g., accuracy, MSE).
    -   `README.md`: Explains that users should add their custom metric scripts here.
-   **`scripts/`**: Contains shell scripts for running experiments.
    -   `train.sh`: An example script to start training. Users should customize paths and arguments.
-   **`utils/`**: Contains various utility functions.
    -   `checkpoint.py`: Handles model and optimizer checkpointing.
    -   `file.py`: File system utilities (listing files, creating directories, saving JSON).
    -   `image.py`: Basic image utilities (denormalization, saving images).
    -   `logger.py`: TensorBoard logger.
    -   `misc.py`: Miscellaneous utilities (datetime, string parsing).
    -   `model.py`: Model-related utilities (e.g., `count_parameters`).
-   **`archive/`**: (Generated) Can be used for storing datasets or cached files.
    -   `cache/`: Default directory for caching data by `utils/file.py` caching functions.
-   **`bin/`**: This directory has been emptied as previous utility scripts were task-specific. Users can add their own command-line tools here if needed.

## Configuration

-   All configuration options are defined in `config.py` using `argparse`.
-   These options can be set via command-line arguments when running `main.py`.
-   Key generic arguments include:
    -   `--exp_id`: Experiment identifier.
    -   `--mode`: `train`, `eval`, or `sample`.
    -   `--train_path`, `--test_path`: Paths to training and testing data.
    -   `--input_shape`: Shape of the input data (e.g., `256 256` for 256x256 images).
    -   `--batch_size`, `--test_batch_size`.
    -   `--lr`: Learning rate.
    -   `--start_iter`, `--end_iter`: Training iteration control.
    -   `--log_every`, `--eval_every`, `--save_every`, `--visualize_every`: Frequencies for various operations.
    -   `--device`: `cuda` or `cpu`.
-   Refer to `config.py` for the full list of available arguments and their default values.

## Getting Started / Usage

1.  **Setup Environment**:
    *   Clone the repository.
    *   Create a Python virtual environment (e.g., using Conda or venv).
    *   Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        *(Note: `requirements.txt` may need review to ensure it matches the generic template's needs.)*

2.  **Prepare Data**:
    *   Organize your dataset into `train` and `test` folders. For image data, `torchvision.datasets.ImageFolder` expects subdirectories for each class within `train_path` and `test_path`. `DefaultDataset` expects a flat list of images.
    *   Update the `TRAIN_DATA_PATH` and `TEST_DATA_PATH` variables in `scripts/train.sh` or provide them as command-line arguments (`--train_path`, `--test_path`). The example `train.sh` creates placeholder directories.

3.  **Define Your Model**:
    *   Open `models/build.py`.
    *   Implement your neural network architecture within the `build_model` function.
    *   Ensure `build_model` returns a `Munch` object containing your model(s). The primary model used by the solver should be accessible via `nets.model`.
        ```python
        # Example in models/build.py
        import torch.nn as nn
        from munch import Munch

        def build_model(args):
            model = nn.Sequential(
                nn.Linear(args.input_shape[0] * args.input_shape[1], 512), # Example for flattened image
                nn.ReLU(),
                nn.Linear(512, 10) # Example for 10 output classes
            )
            nets = Munch(model=model)
            # If you have an EMA model or other auxiliary models, return them here as well.
            # For this generic template, nets_ema is initialized but not actively updated by default.
            nets_ema = Munch(model_ema=None) 
            return nets, nets_ema
        ```

4.  **Define Your Loss Function**:
    *   Open `solver/loss.py`.
    *   Implement your loss calculation logic within the `compute_loss` function.
    *   This function receives `model_output` and `ground_truth` (if applicable) and should return the computed loss (a scalar tensor) and a `Munch` object containing any individual loss components you want to log.
        ```python
        # Example in solver/loss.py
        import torch.nn.functional as F
        from munch import Munch

        def compute_loss(model_output, ground_truth, args):
            # Assuming classification task
            loss = F.cross_entropy(model_output, ground_truth.long())
            loss_items = Munch(cross_entropy=loss.item())
            return loss, loss_items
        ```

5.  **Run Training**:
    *   Modify `scripts/train.sh` to set your desired paths, dataset name, and hyperparameters.
    *   Execute the script from the project root:
        ```bash
        bash scripts/train.sh
        ```
    *   Alternatively, run `main.py` directly with command-line arguments:
        ```bash
        python main.py --mode train --train_path /path/to/your/train_data --test_path /path/to/your/test_data --input_shape H W --lr 0.0001 ...
        ```

6.  **Evaluation and Sampling**:
    *   **Evaluation**: To evaluate a trained model (e.g., calculate average loss on the test set):
        ```bash
        python main.py --mode eval --test_path /path/to/your/test_data --start_iter <iteration_to_load> --exp_id <your_experiment_id>
        ```
    *   **Sampling/Inference**: To run inference with a trained model:
        ```bash
        python main.py --mode sample --test_path /path/to/your/inference_data --start_iter <iteration_to_load> --exp_id <your_experiment_id> --sample_dir <output_directory_for_samples>
        ```
        The `Solver.sample()` method saves raw model outputs. You'll need to customize it or add post-processing to generate specific output formats (e.g., images, text files).

## Customization

To adapt this template for your specific deep learning project:

1.  **Model Architecture (`models/build.py`)**: This is the primary file you'll need to change to define your own neural network(s).
2.  **Loss Function (`solver/loss.py`)**: Define how the loss is calculated based on your model's output and ground truth.
3.  **Data Handling (`data/dataset.py`, `data/loader.py`)**:
    *   If you need custom dataset classes, add them to `dataset.py`.
    *   Modify `loader.py` if you need different data loading strategies or transforms. The current setup uses generic `ImageFolder` and `DefaultDataset` with basic image resizing and normalization.
4.  **Metrics (`metrics/`)**: Add your custom evaluation metric calculations in this directory and integrate them into the `Solver.evaluate()` or `Solver.train()` (evaluation part) methods in `solver/solver.py`.
5.  **Training Script (`scripts/train.sh`)**: Update paths, hyperparameters, and any specific setup for your experiments.
6.  **Configuration (`config.py`)**: Add, remove, or modify command-line arguments as needed for your project.
7.  **Solver Logic (`solver/solver.py`)**:
    *   While the existing loops are generic, you might want to customize aspects of the training (e.g., learning rate schedulers, gradient clipping), visualization, or evaluation.
    *   The `Solver.sample()` method will likely need significant customization to produce meaningful outputs for your task.

## Dependencies

Key dependencies include:
- PyTorch
- torchvision
- tensorboardX (for TensorBoard logging)
- munch
- (Potentially others, check `requirements.txt`)

It's recommended to use a virtual environment (like Conda or venv) to manage project dependencies.

---

*This template was refactored from an earlier GAN-specific version, inspired by StarGAN v2's official implementation. The goal is to provide a cleaner, more generic starting point for diverse PyTorch projects.*
