# PyTorch Deep Learning Template

[English](#english) | [中文](#chinese)

<a id="english"></a>

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

## Running the G3 Model Example

This template now includes an example of how to integrate and run the G3 model (P. Jia et al., 2024), a framework for worldwide geolocalization. The G3 model has its own specific setup and training script, which is called by this template when selected.

### 1. Environment Setup

*   **General Dependencies**: Ensure you have installed all dependencies from the main `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    This includes `accelerate`, `transformers`, and other packages needed by G3.
*   **G3 Specifics**: The G3 model was tested with specific PyTorch versions (torch 2.1.1, torchvision 0.16.1 for CUDA 12.1). Refer to the `models/G3/README.md` for details on setting up a Conda environment tailored for G3 if you encounter issues. The `requirements.txt` has been updated to reflect these versions.

### 2. Data Preparation for G3

The G3 model uses its own datasets and data loading procedures, detailed in `models/G3/README.md`:

*   **Download Datasets**: Follow the instructions in `models/G3/README.md` to download the required datasets (e.g., MP16-Pro). The template's `--train_path` and `--test_path` arguments are not directly used by the G3 training script but are required by the template's configuration.
*   **Location Encoder**: Download the `location_encoder.pth` file (as mentioned in G3's documentation, typically from GeoCLIP) and place it directly inside the `models/G3/` directory. The `run_G3.py` script expects it at `models/G3/location_encoder.pth`.

### 3. Running G3 Training (Geo-Alignment)

*   An example configuration file `g3_train_config.json` is provided in the root directory.
*   To run the G3 training (specifically, the geo-alignment part as described in the G3 paper):
    ```bash
    python main.py g3_train_config.json --mode train
    ```
*   This command will use the template's `main.py` script, which will then detect the `"g3"` model type and execute the `models/G3/run_G3.py` script. The `run_G3.py` script will handle its own training process, including data loading (MP16Dataset) and model checkpointing within the `models/G3/checkpoints/` directory.
*   Output and logs from `run_G3.py` will be printed to the console.

### 4. Other G3 Functionalities

The G3 framework includes other stages like Geo-diversification and Geo-verification. These are **not** currently integrated into the template's `main.py --mode eval` or `--mode sample` flows.

*   To perform these additional steps, please follow the instructions provided in `models/G3/README.md` (e.g., using `IndexSearch.py`, `llm_predict.py`, etc., directly from the `models/G3/` directory).

## Dependencies

Key dependencies include:
- PyTorch
- torchvision
- tensorboardX (for TensorBoard logging)
- munch
- (Potentially others, check `requirements.txt`)

It's recommended to use a virtual environment (like Conda or venv) to manage project dependencies.

---

*This template was refactored from songquanpeng's [pytorch-template](https://github.com/songquanpeng/pytorch-template) template project. The goal is to provide a cleaner, more generic starting point for various PyTorch projects.*

<a id="chinese"></a>

# PyTorch 深度学习模板

## 简介

本项目提供了一个通用且精简的 PyTorch 模板，旨在加速各种深度学习模型的开发。它从之前的 GAN 特定模板重构而来，为广泛的任务（如分类、回归等）提供了一个干净、组织良好的起点。

该模板包括：
- 清晰的项目结构。
- 通过 `config.py` 和命令行参数进行配置管理。
- 在 `solver/solver.py` 中的基本训练、评估和采样循环。
- 在 `models/build.py` 中的模型定义占位符。
- 在 `solver/loss.py` 中的损失计算占位符。
- 用于日志记录（TensorBoard）、检查点保存和文件操作的实用函数。
- 在 `scripts/train.sh` 中的示例训练脚本。

## 项目结构

项目组织为以下主要目录和文件：

-   **`README.md`**: 本文件，提供概述和指南。
-   **`requirements.txt`**: 列出项目依赖项。
-   **`main.py`**: 运行训练、评估或采样的主入口点。
-   **`config.py`**: 定义和管理所有配置选项和参数。
-   **`data/`**: 包含数据加载和预处理工具。
    -   `dataset.py`: 包括用于从平面文件夹加载图像的 `DefaultDataset`。
    -   `loader.py`: 使用通用的 `ImageFolder` 或 `DefaultDataset` 提供 `get_train_loader`、`get_test_loader`、`get_eval_loader`。
    -   `fetcher.py`: 数据加载器的包装器，提供无限迭代器并将数据移动到设备。
-   **`expr/`**: 存储实验输出（日志、模型、样本）的默认目录。
-   **`models/`**: 用于模型架构定义。
    -   `build.py`: **占位符。** 用户必须在此处定义他们的模型。它应该返回一个包含模型的 `Munch` 对象（例如，`nets.model`）。
    -   `README.md`: 简要说明此目录的用途。
-   **`solver/`**: 处理训练、评估和采样逻辑。
    -   `solver.py`: 包含具有通用训练/评估循环的主要 `Solver` 类。
    -   `loss.py`: **占位符。** 包含一个 `compute_loss` 函数，用户必须在其中定义他们的损失计算。
    -   `utils.py`: 求解器的实用工具，如权重初始化。
    -   `misc.py`: 杂项求解器实用工具的占位符。
-   **`metrics/`**: **占位符。** 用于用户定义的度量计算（例如，准确度、MSE）。
    -   `README.md`: 说明用户应在此处添加自定义度量脚本。
-   **`scripts/`**: 包含用于运行实验的 shell 脚本。
    -   `train.sh`: 开始训练的示例脚本。用户应自定义路径和参数。
-   **`utils/`**: 包含各种实用函数。
    -   `checkpoint.py`: 处理模型和优化器检查点。
    -   `file.py`: 文件系统实用工具（列出文件、创建目录、保存 JSON）。
    -   `image.py`: 基本图像实用工具（反归一化、保存图像）。
    -   `logger.py`: TensorBoard 日志记录器。
    -   `misc.py`: 杂项实用工具（日期时间、字符串解析）。
    -   `model.py`: 模型相关实用工具（例如，`count_parameters`）。
-   **`archive/`**: （生成的）可用于存储数据集或缓存文件。
    -   `cache/`: 由 `utils/file.py` 缓存函数缓存数据的默认目录。
-   **`bin/`**: 此目录已清空，因为之前的实用脚本是特定于任务的。用户可以根据需要在此处添加自己的命令行工具。

## 配置

-   所有配置选项都在 `config.py` 中使用 `argparse` 定义。
-   这些选项可以通过运行 `main.py` 时的命令行参数设置。
-   关键通用参数包括：
    -   `--exp_id`: 实验标识符。
    -   `--mode`: `train`、`eval` 或 `sample`。
    -   `--train_path`、`--test_path`: 训练和测试数据的路径。
    -   `--input_shape`: 输入数据的形状（例如，`256 256` 表示 256x256 图像）。
    -   `--batch_size`、`--test_batch_size`。
    -   `--lr`: 学习率。
    -   `--start_iter`、`--end_iter`: 训练迭代控制。
    -   `--log_every`、`--eval_every`、`--save_every`、`--visualize_every`: 各种操作的频率。
    -   `--device`: `cuda` 或 `cpu`。
-   有关可用参数的完整列表及其默认值，请参阅 `config.py`。

## 入门 / 使用方法

1.  **设置环境**:
    *   克隆仓库。
    *   创建 Python 虚拟环境（例如，使用 Conda 或 venv）。
    *   安装依赖项：
        ```bash
        pip install -r requirements.txt
        ```

2.  **准备数据**:
    *   将数据集组织到 `train` 和 `test` 文件夹中。对于图像数据，`torchvision.datasets.ImageFolder` 期望在 `train_path` 和 `test_path` 内为每个类别设置子目录。`DefaultDataset` 期望有一个平面的图像列表。
    *   更新 `scripts/train.sh` 中的 `TRAIN_DATA_PATH` 和 `TEST_DATA_PATH` 变量，或通过命令行参数提供它们（`--train_path`、`--test_path`）。示例 `train.sh` 创建占位符目录。

3.  **定义您的模型**:
    *   打开 `models/build.py`。
    *   在 `build_model` 函数中实现您的神经网络架构。
    *   确保 `build_model` 返回一个包含您的模型的 `Munch` 对象。求解器使用的主要模型应该可以通过 `nets.model` 访问。
        ```python
        # models/build.py 示例
        import torch.nn as nn
        from munch import Munch

        def build_model(args):
            model = nn.Sequential(
                nn.Linear(args.input_shape[0] * args.input_shape[1], 512), # 展平图像的示例
                nn.ReLU(),
                nn.Linear(512, 10) # 10个输出类别的示例
            )
            nets = Munch(model=model)
            # 如果您有 EMA 模型或其他辅助模型，也在此处返回它们。
            # 对于这个通用模板，nets_ema 已初始化但默认情况下不会主动更新。
            nets_ema = Munch(model_ema=None) 
            return nets, nets_ema
        ```

4.  **定义您的损失函数**:
    *   打开 `solver/loss.py`。
    *   在 `compute_loss` 函数中实现您的损失计算逻辑。
    *   此函数接收 `model_output` 和 `ground_truth`（如果适用），并应返回计算的损失（标量张量）和一个包含您想要记录的任何单独损失组件的 `Munch` 对象。
        ```python
        # solver/loss.py 示例
        import torch.nn.functional as F
        from munch import Munch

        def compute_loss(model_output, ground_truth, args):
            # 假设分类任务
            loss = F.cross_entropy(model_output, ground_truth.long())
            loss_items = Munch(cross_entropy=loss.item())
            return loss, loss_items
        ```

5.  **运行训练**:
    *   修改 `scripts/train.sh` 以设置您想要的路径、数据集名称和超参数。
    *   从项目根目录执行脚本：
        ```bash
        bash scripts/train.sh
        ```
    *   或者，直接使用命令行参数运行 `main.py`：
        ```bash
        python main.py --mode train --train_path /path/to/your/train_data --test_path /path/to/your/test_data --input_shape H W --lr 0.0001 ...
        ```

6.  **评估和采样**:
    *   **评估**: 要评估训练好的模型（例如，计算测试集上的平均损失）：
        ```bash
        python main.py --mode eval --test_path /path/to/your/test_data --start_iter <iteration_to_load> --exp_id <your_experiment_id>
        ```
    *   **采样/推理**: 要使用训练好的模型进行推理：
        ```bash
        python main.py --mode sample --test_path /path/to/your/inference_data --start_iter <iteration_to_load> --exp_id <your_experiment_id> --sample_dir <output_directory_for_samples>
        ```
        `Solver.sample()` 方法保存原始模型输出。您需要自定义它或添加后处理以生成特定的输出格式（例如，图像、文本文件）。

## 自定义

要使此模板适应您的特定深度学习项目：

1.  **模型架构 (`models/build.py`)**: 这是您需要更改的主要文件，以定义您自己的神经网络。
2.  **损失函数 (`solver/loss.py`)**: 根据您的模型输出和真实值定义如何计算损失。
3.  **数据处理 (`data/dataset.py`, `data/loader.py`)**:
    *   如果您需要自定义数据集类，请将它们添加到 `dataset.py`。
    *   如果您需要不同的数据加载策略或转换，请修改 `loader.py`。当前设置使用通用的 `ImageFolder` 和 `DefaultDataset`，具有基本的图像调整大小和归一化功能。
4.  **度量 (`metrics/`)**: 在此目录中添加您的自定义评估度量计算，并将它们集成到 `solver/solver.py` 中的 `Solver.evaluate()` 或 `Solver.train()` （评估部分）方法中。
5.  **训练脚本 (`scripts/train.sh`)**: 更新路径、超参数和任何特定于您实验的设置。
6.  **配置 (`config.py`)**: 根据您的项目需要添加、删除或修改命令行参数。
7.  **求解器逻辑 (`solver/solver.py`)**:
    *   虽然现有的循环是通用的，但您可能想要自定义训练的某些方面（例如，学习率调度器、梯度裁剪）、可视化或评估。
    *   `Solver.sample()` 方法可能需要重大自定义，以为您的任务生成有意义的输出。

## Running the G3 Model Example

This template now includes an example of how to integrate and run the G3 model (P. Jia et al., 2024), a framework for worldwide geolocalization. The G3 model has its own specific setup and training script, which is called by this template when selected.

### 1. Environment Setup

*   **General Dependencies**: Ensure you have installed all dependencies from the main `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    This includes `accelerate`, `transformers`, and other packages needed by G3.
*   **G3 Specifics**: The G3 model was tested with specific PyTorch versions (torch 2.1.1, torchvision 0.16.1 for CUDA 12.1). Refer to the `models/G3/README.md` for details on setting up a Conda environment tailored for G3 if you encounter issues. The `requirements.txt` has been updated to reflect these versions.

### 2. Data Preparation for G3

The G3 model uses its own datasets and data loading procedures, detailed in `models/G3/README.md`:

*   **Download Datasets**: Follow the instructions in `models/G3/README.md` to download the required datasets (e.g., MP16-Pro). The template's `--train_path` and `--test_path` arguments are not directly used by the G3 training script but are required by the template's configuration.
*   **Location Encoder**: Download the `location_encoder.pth` file (as mentioned in G3's documentation, typically from GeoCLIP) and place it directly inside the `models/G3/` directory. The `run_G3.py` script expects it at `models/G3/location_encoder.pth`.

### 3. Running G3 Training (Geo-Alignment)

*   An example configuration file `g3_train_config.json` is provided in the root directory.
*   To run the G3 training (specifically, the geo-alignment part as described in the G3 paper):
    ```bash
    python main.py g3_train_config.json --mode train
    ```
*   This command will use the template's `main.py` script, which will then detect the `"g3"` model type and execute the `models/G3/run_G3.py` script. The `run_G3.py` script will handle its own training process, including data loading (MP16Dataset) and model checkpointing within the `models/G3/checkpoints/` directory.
*   Output and logs from `run_G3.py` will be printed to the console.

### 4. Other G3 Functionalities

The G3 framework includes other stages like Geo-diversification and Geo-verification. These are **not** currently integrated into the template's `main.py --mode eval` or `--mode sample` flows.

*   To perform these additional steps, please follow the instructions provided in `models/G3/README.md` (e.g., using `IndexSearch.py`, `llm_predict.py`, etc., directly from the `models/G3/` directory).

## 依赖项

主要依赖项包括：
- PyTorch
- torchvision
- tensorboardX（用于 TensorBoard 日志记录）
- munch
- （可能还有其他，请查看 `requirements.txt`）

建议使用虚拟环境（如 Conda 或 venv）来管理项目依赖项。

---

*此模板从songquanpeng的[pytorch-template](https://github.com/songquanpeng/pytorch-template)模板项目重构而来。目标是为各种 PyTorch 项目提供一个更干净、更通用的起点。*
