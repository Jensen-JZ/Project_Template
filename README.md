# Project Title: Advanced Geolocation Model Framework

*[English](#english-version) | [中文](#chinese-version)*

<a name="english-version"></a>
## Overview

This project provides a comprehensive framework for developing, training, and evaluating advanced geolocation models, with a primary focus on the G3 model architecture. It includes modules for data handling, model definitions, training pipelines, feature extraction, geospatial operations, LLM-based prediction services, and results aggregation. The structure is designed to be modular, allowing for easy extension and experimentation with new models, data sources, and evaluation techniques.

## Directory Structure

The project is organized into the following key directories:

*   **`models/`**: Contains definitions for core neural network architectures.
    *   `G3.py`: Defines the G3 geolocalization model.
    *   `README.md`: Details on models and how to add new ones.
*   **`data/`**: Manages datasets and data loading utilities.
    *   `dataset.py`, `loader.py`, `fetcher.py`: Modules for dataset definition, loading, and fetching.
    *   `README.md`: Explains data organization and module usage.
*   **`dataprocessing/`**: Houses modules for data transformation, aggregation, and (post-)processing.
    *   `g3_prediction_aggregation.py`: Aggregates LLM prediction results.
    *   `README.md`: Details on data processing modules.
*   **`features/`**: For modules performing complex feature extraction or specialized model operations.
    *   `g3_geospatial_operations.py`: Handles FAISS indexing, searching, and evaluation using G3 model embeddings.
    *   `README.md`: Information on feature-related modules.
*   **`services/`**: Encapsulates complex functionalities, especially those interacting with external APIs or managing distinct operational logic.
    *   `llm_geolocation.py`: Provides classes for geolocation using LLMs (OpenAI and local Hugging Face models).
    *   `README.md`: Describes available services.
*   **`training/`**: Contains modules for defining and managing model training processes.
    *   `g3_trainer.py`: Defines the `G3Trainer` class for training the G3 model.
    *   `README.md`: Explains training components and how to add new trainers.
*   **`solver/`**: Manages core training logic, including loss functions and optimization procedures. (Note: Much of this might be integrated into `training/g3_trainer.py` now).
    *   `loss.py`, `solver.py`, `misc.py`, `utils.py`: Components for the training loop.
    *   `README.md`: Details on solver components.
*   **`scripts/`**: Contains launcher scripts for various workflows (training, evaluation, data processing, etc.).
    *   `execute_*.py`: Launcher scripts for the refactored modules.
    *   `train.sh`: Example shell script for training.
    *   `README.md`: Explains how to use the launcher scripts.
*   **`utils/`**: A collection of general utility functions and helper modules.
    *   Includes modules for checkpointing, file operations, image processing, logging, etc.
    *   `rff/`: Subdirectory for Random Fourier Features implementation.
    *   `README.md`: Describes available utilities.
*   **`metrics/`**: For defining and computing evaluation metrics.
    *   `README.md`: Guidelines on adding and using metrics.
*   **`examples/`**: Contains example scripts demonstrating usage of core components.
    *   `run_g3_example.py`: Shows how to run the G3 model with dummy inputs.
*   **`archive/`**: (As seen in `ls` output) Likely for storing older or deprecated project components.
*   `config.py`: Defines and manages configuration options (via `argparse`) for the generic template framework used by `main.py`. **Note**: While some general settings might be adaptable, G3-specific operations launched via `scripts/` typically manage their own configurations directly. Refer to individual scripts in `scripts/` for their specific arguments.
*   `main.py`: Entry point for the original generic PyTorch template. It uses `config.py` and `solver/solver.py` for a general-purpose training/evaluation loop. **Note**: For G3-specific model operations (training, indexing, etc.), please use the launcher scripts in the `scripts/` directory.
*   `requirements.txt`: Lists project dependencies.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a Python environment:**
    It's recommended to use a virtual environment (e.g., venv, conda).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up PYTHONPATH (if needed):**
    Most scripts are structured to run correctly from the project root. If you encounter import errors, ensure the project root directory is in your `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root
    ```

## Usage

This project uses a modular structure where core logic is implemented in directories like `training/`, `features/`, `services/`, and `dataprocessing/`. These functionalities are typically invoked via launcher scripts located in the `scripts/` directory.

1.  **Explore Launcher Scripts**: Navigate to the `scripts/` directory. The `scripts/README.md` provides an overview of available launcher scripts and their purposes.
2.  **Get Help**: Each Python launcher script supports a `--help` argument that lists its specific command-line options:
    ```bash
    python scripts/<launcher_script_name>.py --help
    ```
    For example:
    ```bash
    python scripts/execute_g3_training.py --help
    python scripts/execute_g3_geo_ops.py --help
    python scripts/execute_openai_geoloc.py --help
    ```
3.  **Running Workflows**:
    *   **Training**: Use `scripts/execute_g3_training.py` to train the G3 model. Configure parameters as needed via command-line arguments. The `scripts/train.sh` script provides an example of how to run this.
    *   **Geospatial Operations (Indexing, Searching, Evaluation)**: Use `scripts/execute_g3_geo_ops.py`.
    *   **LLM Geolocation (OpenAI)**: Use `scripts/execute_openai_geoloc.py`. Requires API keys and appropriate configuration.
    *   **LLM Geolocation (Hugging Face LLaVA)**: Use `scripts/execute_hf_llava_geoloc.py`. Requires a local LLaVA model.
    *   **Prediction Aggregation**: Use `scripts/execute_g3_aggregation.py` to combine various prediction outputs.

Refer to the `README.md` files within each specific directory (`models/`, `data/`, `training/`, etc.) for more detailed information on their respective components and how to extend them.

## Adding New Components

The project is designed for modularity. When adding new functionalities:

*   **New Models**: Add to `models/` and update `models/README.md`. Consider if `models/build.py` needs changes.
*   **New Training Logic**: Create new trainers in `training/` and update `training/README.md`. Add a corresponding launcher in `scripts/`.
*   **New Feature/Geospatial Operations**: Add to `features/` and update `features/README.md`. Add a launcher in `scripts/`.
*   **New Services (e.g., API clients, complex pipelines)**: Add to `services/` and update `services/README.md`. Add a launcher in `scripts/`.
*   **New Data Processing Steps**: Add to `dataprocessing/` and update `dataprocessing/README.md`. Add a launcher in `scripts/`.
*   **New Datasets/Loaders**: Add definitions to `data/dataset.py` or `data/loader.py` and update `data/README.md`.
*   **General Utilities**: Add to appropriate modules in `utils/` and update `utils/README.md`.

Always create or update relevant README files when adding new components or directories.

---

<a name="chinese-version"></a>
# 项目标题：高级地理位置模型框架

## 概述

本项目提供了一个全面的框架，用于开发、训练和评估高级地理位置模型，主要关注G3模型架构。它包括数据处理、模型定义、训练流程、特征提取、地理空间操作、基于LLM的预测服务和结果聚合等模块。该结构设计为模块化，允许轻松扩展和实验新模型、数据源和评估技术。

## 目录结构

项目组织为以下主要目录：

*   **`models/`**: 包含核心神经网络架构的定义。
    *   `G3.py`: 定义G3地理定位模型。
    *   `README.md`: 关于模型及如何添加新模型的详细信息。
*   **`data/`**: 管理数据集和数据加载工具。
    *   `dataset.py`, `loader.py`, `fetcher.py`: 用于数据集定义、加载和获取的模块。
    *   `README.md`: 解释数据组织和模块使用方法。
*   **`dataprocessing/`**: 包含数据转换、聚合和（后）处理的模块。
    *   `g3_prediction_aggregation.py`: 聚合LLM预测结果。
    *   `README.md`: 关于数据处理模块的详细信息。
*   **`features/`**: 用于执行复杂特征提取或专门模型操作的模块。
    *   `g3_geospatial_operations.py`: 使用G3模型嵌入处理FAISS索引、搜索和评估。
    *   `README.md`: 关于特征相关模块的信息。
*   **`services/`**: 封装复杂功能，特别是与外部API交互或管理不同操作逻辑的功能。
    *   `llm_geolocation.py`: 提供使用LLM（OpenAI和本地Hugging Face模型）进行地理定位的类。
    *   `README.md`: 描述可用服务。
*   **`training/`**: 包含定义和管理模型训练过程的模块。
    *   `g3_trainer.py`: 定义用于训练G3模型的`G3Trainer`类。
    *   `README.md`: 解释训练组件和如何添加新训练器。
*   **`solver/`**: 管理核心训练逻辑，包括损失函数和优化程序。（注意：很多内容可能已集成到`training/g3_trainer.py`中）
    *   `loss.py`, `solver.py`, `misc.py`, `utils.py`: 训练循环的组件。
    *   `README.md`: 关于求解器组件的详细信息。
*   **`scripts/`**: 包含各种工作流程（训练、评估、数据处理等）的启动脚本。
    *   `execute_*.py`: 重构模块的启动脚本。
    *   `train.sh`: 训练的示例shell脚本。
    *   `README.md`: 解释如何使用启动脚本。
*   **`utils/`**: 通用工具函数和辅助模块的集合。
    *   包括用于检查点、文件操作、图像处理、日志记录等的模块。
    *   `rff/`: 随机傅里叶特征实现的子目录。
    *   `README.md`: 描述可用工具。
*   **`metrics/`**: 用于定义和计算评估指标。
    *   `README.md`: 添加和使用指标的指南。
*   **`examples/`**: 包含演示核心组件使用的示例脚本。
    *   `run_g3_example.py`: 展示如何使用伪输入运行G3模型。
*   **`archive/`**: （如`ls`输出所示）可能用于存储较旧或不推荐使用的项目组件。
*   `config.py`: 定义和管理`main.py`使用的通用模板框架的配置选项（通过`argparse`）。**注意**：虽然一些通用设置可能是可适应的，但通过`scripts/`启动的G3特定操作通常直接管理其自己的配置。有关特定参数，请参考`scripts/`中的各个脚本。
*   `main.py`: 原始通用PyTorch模板的入口点。它使用`config.py`和`solver/solver.py`进行通用训练/评估循环。**注意**：对于G3特定的模型操作（训练、索引等），请使用`scripts/`目录中的启动脚本。
*   `requirements.txt`: 列出项目依赖项。

## 设置和安装

1.  **克隆仓库：**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **创建Python环境：**
    建议使用虚拟环境（例如venv、conda）。
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows上：venv\Scripts\activate
    ```
3.  **安装依赖：**
    ```bash
    pip install -r requirements.txt
    ```
4.  **设置PYTHONPATH（如需要）：**
    大多数脚本都设计为从项目根目录正确运行。如果遇到导入错误，请确保项目根目录在您的`PYTHONPATH`中：
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root
    ```

## 使用方法

该项目使用模块化结构，核心逻辑在`training/`、`features/`、`services/`和`dataprocessing/`等目录中实现。这些功能通常通过位于`scripts/`目录中的启动脚本调用。

1.  **探索启动脚本**：导航到`scripts/`目录。`scripts/README.md`提供了可用启动脚本及其用途的概述。
2.  **获取帮助**：每个Python启动脚本都支持`--help`参数，列出其特定的命令行选项：
    ```bash
    python scripts/<launcher_script_name>.py --help
    ```
    例如：
    ```bash
    python scripts/execute_g3_training.py --help
    python scripts/execute_g3_geo_ops.py --help
    python scripts/execute_openai_geoloc.py --help
    ```
3.  **运行工作流程**：
    *   **训练**：使用`scripts/execute_g3_training.py`训练G3模型。根据需要通过命令行参数配置参数。`scripts/train.sh`脚本提供了如何运行此操作的示例。
    *   **地理空间操作（索引、搜索、评估）**：使用`scripts/execute_g3_geo_ops.py`。
    *   **LLM地理定位（OpenAI）**：使用`scripts/execute_openai_geoloc.py`。需要API密钥和适当的配置。
    *   **LLM地理定位（Hugging Face LLaVA）**：使用`scripts/execute_hf_llava_geoloc.py`。需要本地LLaVA模型。
    *   **预测聚合**：使用`scripts/execute_g3_aggregation.py`组合各种预测输出。

有关各自组件的更详细信息以及如何扩展它们的信息，请参阅每个特定目录（`models/`、`data/`、`training/`等）中的`README.md`文件。

## 添加新组件

该项目设计为模块化。添加新功能时：

*   **新模型**：添加到`models/`并更新`models/README.md`。考虑`models/build.py`是否需要更改。
*   **新训练逻辑**：在`training/`中创建新的训练器并更新`training/README.md`。在`scripts/`中添加相应的启动器。
*   **新特征/地理空间操作**：添加到`features/`并更新`features/README.md`。在`scripts/`中添加启动器。
*   **新服务（例如，API客户端、复杂管道）**：添加到`services/`并更新`services/README.md`。在`scripts/`中添加启动器。
*   **新数据处理步骤**：添加到`dataprocessing/`并更新`dataprocessing/README.md`。在`scripts/`中添加启动器。
*   **新数据集/加载器**：在`data/dataset.py`或`data/loader.py`中添加定义，并更新`data/README.md`。
*   **通用工具**：添加到`utils/`中的适当模块，并更新`utils/README.md`。

添加新组件或目录时，始终创建或更新相关的README文件。
