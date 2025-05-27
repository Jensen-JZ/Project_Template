# Scripts Directory

[English](#english) | [中文](#chinese)

<a id="english"></a>
## Purpose

The `scripts/` directory serves as the primary location for launching various workflows and operations within this project. Most Python scripts in this directory are **launchers** that execute core logic implemented in other dedicated modules (e.g., `training/`, `features/`, `services/`, `dataprocessing/`). This separation helps in organizing the codebase and making the core functionalities reusable.

## Current Structure & Launcher Script Descriptions

This directory currently contains the following launcher scripts:

*   **`execute_g3_training.py`**:
    *   **Purpose**: Launches the training process for the G3 model.
    *   **Core Logic**: Implemented in the `G3Trainer` class within `training/g3_trainer.py`. This script handles parsing training configurations and initiating the trainer.
*   **`execute_g3_geo_ops.py`**:
    *   **Purpose**: Executes various geospatial operations related to the G3 model, such as building FAISS indices from model embeddings, searching these indices, and evaluating search results.
    *   **Core Logic**: Implemented in modules within `features/g3_geospatial_operations.py`.
*   **`execute_openai_geoloc.py`**:
    *   **Purpose**: Runs geolocation prediction tasks using Large Language Models (LLMs) via the OpenAI API. This includes zero-shot predictions and RAG-based predictions.
    *   **Core Logic**: Implemented in the `OpenAIGeolocator` class within `services/llm_geolocation.py`.
*   **`execute_hf_llava_geoloc.py`**:
    *   **Purpose**: Runs geolocation prediction tasks using local Hugging Face LLaVA (or similar compatible) models. Supports zero-shot and RAG-based predictions.
    *   **Core Logic**: Implemented in the `HuggingFaceLlavaGeolocator` class within `services/llm_geolocation.py`.
*   **`execute_g3_aggregation.py`**:
    *   **Purpose**: Aggregates and consolidates LLM prediction results from various sources or runs into a final structured format.
    *   **Core Logic**: Implemented in the `aggregate_llm_predictions` function within `dataprocessing/g3_prediction_aggregation.py`.
*   **`train.sh`**:
    *   **Purpose**: An example shell script that demonstrates how to execute a training process. It likely calls one of the Python launcher scripts (e.g., `execute_g3_training.py`) with specific command-line arguments and environment configurations. This script can be adapted for specific training setups or batch job submissions.

## Usage

All Python launcher scripts in this directory are designed to be run from the command line. They accept various arguments to control their behavior, such as paths to data, model configurations, API keys, etc.

**General Usage Pattern:**

To understand the specific arguments and options available for each launcher script, use the `--help` flag:

```bash
python scripts/<launcher_script_name>.py --help
```
For example:
```bash
python scripts/execute_g3_training.py --help
python scripts/execute_openai_geoloc.py --help
```

**Executing Scripts:**

Once you know the required arguments, you can run the scripts as follows:

```bash
python scripts/<launcher_script_name>.py --argument1 value1 --argument2 value2 ...
```

*   **`train.sh`**:
    *   This is an executable shell script.
    *   Usage: `./scripts/train.sh`
    *   You may need to modify this script internally to set specific parameters or paths accordingto your environment and which Python launcher it invokes. Ensure it has execute permissions (`chmod +x scripts/train.sh`).

**Environment Setup:**

*   Ensure all dependencies listed in `requirements.txt` are installed in your Python environment.
*   Some scripts might require specific environment variables to be set (e.g., API keys for LLM services, `CUDA_VISIBLE_DEVICES`). Refer to the help output of each script or the documentation of the underlying core modules.
*   Ensure that your `PYTHONPATH` is set up correctly so that modules within the project (e.g., from `models/`, `utils/`, `training/`, `features/`, `services/`, `dataprocessing/`) can be imported by the scripts. Typically, running scripts from the project's root directory handles this, or you might need to add the project root to `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root
    ```

## Adding New Launcher Scripts

1.  **Develop Core Logic First**: Implement the primary functionality in a dedicated module (e.g., in `services/`, `features/`, `dataprocessing/`, or a new directory).
2.  **Create the Launcher Script**: Add a new Python script in this `scripts/` directory.
3.  **Implement Argument Parsing**: Use `argparse` to define command-line arguments that your launcher script will accept. These arguments should configure and control the core logic.
4.  **Call Core Logic**: In the launcher script, import the necessary functions or classes from your core module and execute them using the parsed arguments.
5.  **Document the Launcher**: Update this `README.md` file:
    *   Add your new launcher script to the list under "Current Structure & Launcher Script Descriptions".
    *   Provide a brief description of its purpose and point to the module where the core logic resides.
    *   Mention any critical or new usage patterns.
6.  **Make Shell Scripts Executable**: If you add a new shell script, grant it execute permissions (e.g., `chmod +x scripts/my_new_launcher.sh`).

<a id="chinese"></a>
# 脚本目录

## 目的

`scripts/` 目录是本项目中用于启动各种工作流程和操作的主要位置。该目录中的大多数Python脚本都是**启动器**，用于执行在其他专用模块（例如 `training/`、`features/`、`services/`、`dataprocessing/`）中实现的核心逻辑。这种分离有助于组织代码库并使核心功能可重复使用。

## 当前结构与启动器脚本说明

本目录当前包含以下启动器脚本：

*   **`execute_g3_training.py`**：
    *   **目的**：启动G3模型的训练过程。
    *   **核心逻辑**：在 `training/g3_trainer.py` 的 `G3Trainer` 类中实现。此脚本处理训练配置的解析并启动训练器。
*   **`execute_g3_geo_ops.py`**：
    *   **目的**：执行与G3模型相关的各种地理空间操作，如从模型嵌入构建FAISS索引、搜索这些索引以及评估搜索结果。
    *   **核心逻辑**：在 `features/g3_geospatial_operations.py` 的模块中实现。
*   **`execute_openai_geoloc.py`**：
    *   **目的**：通过OpenAI API使用大型语言模型（LLM）运行地理定位预测任务。这包括零样本预测和基于RAG的预测。
    *   **核心逻辑**：在 `services/llm_geolocation.py` 的 `OpenAIGeolocator` 类中实现。
*   **`execute_hf_llava_geoloc.py`**：
    *   **目的**：使用本地Hugging Face LLaVA（或类似兼容的）模型运行地理定位预测任务。支持零样本和基于RAG的预测。
    *   **核心逻辑**：在 `services/llm_geolocation.py` 的 `HuggingFaceLlavaGeolocator` 类中实现。
*   **`execute_g3_aggregation.py`**：
    *   **目的**：将来自各种来源或运行的LLM预测结果聚合并整合为最终结构化格式。
    *   **核心逻辑**：在 `dataprocessing/g3_prediction_aggregation.py` 的 `aggregate_llm_predictions` 函数中实现。
*   **`train.sh`**：
    *   **目的**：一个示例shell脚本，展示如何执行训练过程。它可能调用其中一个Python启动器脚本（例如 `execute_g3_training.py`）并附带特定的命令行参数和环境配置。该脚本可以针对特定的训练设置或批处理作业提交进行调整。

## 使用方法

本目录中的所有Python启动器脚本都设计为从命令行运行。它们接受各种参数来控制其行为，例如数据路径、模型配置、API密钥等。

**一般使用模式：**

要了解每个启动器脚本可用的具体参数和选项，请使用 `--help` 标志：

```bash
python scripts/<启动器脚本名称>.py --help
```
例如：
```bash
python scripts/execute_g3_training.py --help
python scripts/execute_openai_geoloc.py --help
```

**执行脚本：**

一旦了解了所需的参数，您可以按如下方式运行脚本：

```bash
python scripts/<启动器脚本名称>.py --参数1 值1 --参数2 值2 ...
```

*   **`train.sh`**：
    *   这是一个可执行的shell脚本。
    *   用法：`./scripts/train.sh`
    *   您可能需要根据您的环境和它调用的Python启动器，在脚本内部修改特定的参数或路径。确保它具有执行权限（`chmod +x scripts/train.sh`）。

**环境设置：**

*   确保在您的Python环境中安装了 `requirements.txt` 中列出的所有依赖项。
*   某些脚本可能需要设置特定的环境变量（例如，LLM服务的API密钥、`CUDA_VISIBLE_DEVICES`）。请参考每个脚本的帮助输出或底层核心模块的文档。
*   确保您的 `PYTHONPATH` 设置正确，以便脚本可以导入项目内的模块（例如，来自 `models/`、`utils/`、`training/`、`features/`、`services/`、`dataprocessing/`）。通常，从项目的根目录运行脚本会处理这个问题，或者您可能需要将项目根目录添加到 `PYTHONPATH`：
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root
    ```

## 添加新的启动器脚本

1.  **首先开发核心逻辑**：在专用模块中实现主要功能（例如，在 `services/`、`features/`、`dataprocessing/` 或新目录中）。
2.  **创建启动器脚本**：在这个 `scripts/` 目录中添加新的Python脚本。
3.  **实现参数解析**：使用 `argparse` 定义您的启动器脚本将接受的命令行参数。这些参数应该配置和控制核心逻辑。
4.  **调用核心逻辑**：在启动器脚本中，从您的核心模块导入必要的函数或类，并使用解析的参数执行它们。
5.  **记录启动器**：更新这个 `README.md` 文件：
    *   将您的新启动器脚本添加到"当前结构与启动器脚本说明"下的列表中。
    *   提供其目的的简要说明，并指向核心逻辑所在的模块。
    *   提及任何关键或新的使用模式。
6.  **使Shell脚本可执行**：如果您添加了新的shell脚本，请授予其执行权限（例如，`chmod +x scripts/my_new_launcher.sh`）。
