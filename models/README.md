# Models Directory

[English](#english) | [中文](#chinese)

<a name="english"></a>
## Purpose

The `models/` directory serves as a central repository for all model definitions used within this project. Each primary Python file in this directory (or within its subdirectories, if any are added in the future) should define a distinct model architecture or a related set of model utilities.

## Current Structure

Currently, the `models/` directory contains the following key files:

*   `G3.py`: Defines the G3 model, a neural network architecture specifically designed for geolocalization tasks. It typically includes image encoders, text encoders, and location encoders, along with mechanisms to combine their features for predicting geographical coordinates from multimedia inputs.
*   `build.py`: Contains helper functions or classes responsible for instantiating model objects. This script might include logic to select a model based on configuration, load pretrained weights, or set up model-specific parameters.
*   `README.md`: This file, providing an overview of the `models/` directory.

## Adding New Models

To add a new model to this directory:

1.  **Create a new Python file**: Name it descriptively (e.g., `my_new_model.py`).
2.  **Define your model class**: Implement your model architecture within this file. Ensure it's well-documented.
3.  **Update `build.py` (if applicable)**: If your model requires specific instantiation logic or needs to be selectable through a centralized builder function, update `build.py` to include your new model. This might involve adding a new function or modifying an existing one to recognize and construct your model.
4.  **Add unit tests**: It's highly recommended to add unit tests for your new model in the appropriate testing directory to ensure its correctness and facilitate maintenance.
5.  **Update this README**: Briefly describe your new model file under the "Current Structure" section or create a new section if the model is significantly different.

## Usage Examples

The primary usage of these models is typically within training, evaluation, or inference scripts. For detailed examples of how to instantiate and use these models, please refer to the scripts located in the `scripts/` directory.

For example, to use the `G3` model:

```python
# (Illustrative example - actual usage might vary based on scripts/run_g3.py or similar)
# Ensure your PYTHONPATH is set up correctly if running from outside the project root

from models.G3 import G3
from models.build import build_model # Assuming build_model can construct G3

# Direct instantiation (if applicable)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# g3_model = G3(device=device)

# Or, using a builder function from build.py (preferred if available)
# model_config = {"name": "G3", "params": {...}}
# g3_model = build_model(model_config)

# See scripts like scripts/run_g3.py for actual training and data loading examples.
```

Refer to specific training scripts (e.g., `scripts/run_g3.py`) for comprehensive examples of data loading, model training, and inference workflows.

<a name="chinese"></a>
# 模型目录

## 目的

`models/` 目录作为本项目中所有模型定义的中央存储库。该目录中的每个主要Python文件（或未来可能添加的子目录中的文件）都应该定义一个独特的模型架构或相关的模型工具集。

## 当前结构

目前，`models/` 目录包含以下关键文件：

* `G3.py`：定义G3模型，这是一个专为地理定位任务设计的神经网络架构。它通常包括图像编码器、文本编码器和位置编码器，以及将它们的特征组合起来以从多媒体输入预测地理坐标的机制。
* `build.py`：包含负责实例化模型对象的辅助函数或类。该脚本可能包括基于配置选择模型、加载预训练权重或设置模型特定参数的逻辑。
* `README.md`：本文件，提供`models/`目录的概述。

## 添加新模型

要向此目录添加新模型：

1. **创建新的Python文件**：为其命名一个描述性名称（例如，`my_new_model.py`）。
2. **定义你的模型类**：在此文件中实现你的模型架构。确保它有良好的文档。
3. **更新`build.py`（如适用）**：如果你的模型需要特定的实例化逻辑或需要通过集中式构建函数选择，请更新`build.py`以包含你的新模型。这可能涉及添加新函数或修改现有函数以识别和构建你的模型。
4. **添加单元测试**：强烈建议在适当的测试目录中为你的新模型添加单元测试，以确保其正确性并促进维护。
5. **更新此README**：在"当前结构"部分下简要描述你的新模型文件，或者如果该模型与众不同，则创建一个新部分。

## 使用示例

这些模型的主要用途通常在训练、评估或推理脚本中。有关如何实例化和使用这些模型的详细示例，请参考位于`scripts/`目录中的脚本。

例如，要使用`G3`模型：

```python
# （说明性示例 - 实际使用可能因scripts/run_g3.py或类似脚本而异）
# 如果从项目根目录外运行，请确保正确设置PYTHONPATH

from models.G3 import G3
from models.build import build_model # 假设build_model可以构建G3

# 直接实例化（如适用）
# device = "cuda" if torch.cuda.is_available() else "cpu"
# g3_model = G3(device=device)

# 或者，使用build.py中的构建函数（如有可用，首选）
# model_config = {"name": "G3", "params": {...}}
# g3_model = build_model(model_config)

# 有关实际训练和数据加载示例，请参见脚本，如scripts/run_g3.py。
```

有关数据加载、模型训练和推理工作流程的完整示例，请参考特定的训练脚本（例如，`scripts/run_g3.py`）。
