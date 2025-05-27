# Data Directory

[English](#english) | [中文](#chinese)

<a id="english"></a>
## Purpose

The `data/` directory is responsible for housing all data-related aspects of this project. This includes:

*   **Raw Datasets**: Subdirectories for storing raw datasets (e.g., image files, CSVs with metadata). (Currently, no raw dataset subdirectories are directly listed at the top level of `data/`, but they would be placed here, like `data/im2gps3k/` or `data/yfcc4k/` which are referenced by scripts).
*   **Preprocessed Data**: Any intermediate or preprocessed data files generated from raw datasets.
*   **Data-related Modules**: Python scripts that define how data is handled, loaded, and processed.

## Current Structure

The `data/` directory currently contains the following key Python modules:

*   **`dataset.py`**:
    *   **Description**: This module is central to defining how datasets are structured and accessed. It likely contains PyTorch `Dataset` classes (e.g., for datasets like MP16, im2gps3k, yfcc4k previously defined in `utils/g3_utils.py` and potentially moved or referenced here). These classes handle the logic for loading individual data samples (images, texts, coordinates) and applying initial transformations.
*   **`fetcher.py`**:
    *   **Description**: This module's purpose is likely to fetch or download datasets from external sources. It might contain functions or scripts to retrieve raw data files if they are not already present locally.
*   **`loader.py`**:
    *   **Description**: This module focuses on data loading utilities. It would typically contain functions to create PyTorch `DataLoader` instances, which handle batching, shuffling, and parallel data loading for efficient model training and evaluation. It might also include data augmentation pipelines or further preprocessing steps applied at the batch level.
*   **`README.md`**: This file, providing an overview of the `data/` directory.

## Data Organization

When adding new datasets to the project, follow these guidelines:

1.  **Create a Subdirectory**: For each new raw dataset, create a dedicated subdirectory within `data/`. For example, if adding the "MyNewDataset", create `data/MyNewDataset/`.
    *   Store raw images, metadata files (CSVs, JSONs), and any other original dataset components within this subdirectory.
2.  **Update `.gitignore`**: If the raw data is large, consider adding the specific dataset subdirectory to the project's `.gitignore` file to avoid committing large data files to the repository. Instead, provide download instructions or use the `fetcher.py` module.
3.  **Preprocessing Scripts**: If the dataset requires preprocessing, any scripts used for this should ideally be placed in the `scripts/` directory, with clear instructions on how to run them. Preprocessed output might be stored in the dataset's subdirectory or a new `data/processed/<DatasetName>/` directory.
4.  **Dataset Class**: Implement a new PyTorch `Dataset` class for your dataset in `dataset.py` (or a new, appropriately named module if `dataset.py` becomes too large).
5.  **Loader Functions**: Add or update functions in `loader.py` to create `DataLoader` instances for your new dataset.
6.  **Update this README**: Document the new dataset subdirectory and any specific organizational details.

## Usage Instructions

The modules within this directory are used by the training, evaluation, and processing scripts found in `scripts/`.

*   **Preparing Data**:
    1.  **Fetch/Place Data**: Ensure the required raw datasets are present. This might involve running a script using `fetcher.py` or manually downloading and placing data into the appropriate subdirectories (e.g., `data/im2gps3k/images/`, `data/MP16_Pro_filtered.csv`).
    2.  **Preprocessing**: If any preprocessing steps are necessary (as defined by scripts or documentation), run them.
*   **Using Data Modules**:
    *   The `Dataset` classes in `dataset.py` are typically instantiated by providing paths to the data and any necessary processors (e.g., image or text processors from a model).
    *   The `DataLoader` functions in `loader.py` are then used to wrap these `Dataset` objects for use in model training.

**Example Workflow (Conceptual):**

```python
# In a script (e.g., scripts/run_g3.py)

from data.dataset import MP16Dataset # Assuming MP16Dataset is defined here
from data.loader import create_mp16_dataloader # Assuming a helper function

# Configuration (paths, batch size, etc.)
data_root = "./data/mp16/" # Path to MP16 data
csv_path = "./data/MP16_Pro_filtered.csv"
image_archive_path = "./data/mp-16-images.tar"
batch_size = 64
# vision_processor and text_processor would come from the model

# 1. Instantiate the Dataset
# mp16_dataset = MP16Dataset(root_path=data_root, text_data_path=csv_path, ...)
# (Note: Actual MP16Dataset, im2gps3kDataset, yfcc4kDataset were previously in utils.g3_utils.py;
#  this example assumes they or similar classes would be defined/managed by data/dataset.py)

# 2. Create a DataLoader
# train_dataloader = create_mp16_dataloader(mp16_dataset, batch_size=batch_size, ...)

# 3. Use in training loop
# for batch in train_dataloader:
#     images, texts, coordinates = batch
#     # ... process batch ...
```

Refer to the specific scripts in the `scripts/` directory for concrete examples of how these data modules are instantiated and utilized.

<a id="chinese"></a>
# 数据目录

## 目的

`data/` 目录负责存放该项目的所有数据相关内容，包括：

*   **原始数据集**：存储原始数据集的子目录（例如，图像文件、带有元数据的CSV文件）。（目前，在 `data/` 的顶层没有直接列出原始数据集子目录，但它们会被放置在这里，如脚本中引用的 `data/im2gps3k/` 或 `data/yfcc4k/`）。
*   **预处理数据**：从原始数据集生成的任何中间或预处理数据文件。
*   **数据相关模块**：定义数据如何处理、加载和处理的Python脚本。

## 当前结构

`data/` 目录目前包含以下关键Python模块：

*   **`dataset.py`**：
    *   **描述**：该模块是定义数据集结构和访问方式的核心。它可能包含PyTorch的 `Dataset` 类（例如，用于之前在 `utils/g3_utils.py` 中定义并可能移动或引用到这里的MP16、im2gps3k、yfcc4k等数据集）。这些类处理加载单个数据样本（图像、文本、坐标）和应用初始转换的逻辑。
*   **`fetcher.py`**：
    *   **描述**：该模块的目的可能是从外部源获取或下载数据集。它可能包含函数或脚本，用于在本地不存在原始数据文件的情况下检索它们。
*   **`loader.py`**：
    *   **描述**：该模块专注于数据加载实用工具。它通常包含用于创建PyTorch `DataLoader` 实例的函数，这些实例处理批处理、洗牌和并行数据加载，以便高效地进行模型训练和评估。它还可能包括数据增强管道或在批处理级别应用的进一步预处理步骤。
*   **`README.md`**：本文件，提供 `data/` 目录的概述。

## 数据组织

在向项目添加新数据集时，请遵循以下指南：

1.  **创建子目录**：对于每个新的原始数据集，在 `data/` 中创建一个专用子目录。例如，如果添加"MyNewDataset"，创建 `data/MyNewDataset/`。
    *   在此子目录中存储原始图像、元数据文件（CSV、JSON）和任何其他原始数据集组件。
2.  **更新 `.gitignore`**：如果原始数据很大，考虑将特定数据集子目录添加到项目的 `.gitignore` 文件中，以避免将大型数据文件提交到仓库。相反，提供下载说明或使用 `fetcher.py` 模块。
3.  **预处理脚本**：如果数据集需要预处理，任何用于此目的的脚本理想情况下应放置在 `scripts/` 目录中，并提供清晰的运行说明。预处理输出可能存储在数据集的子目录或新的 `data/processed/<DatasetName>/` 目录中。
4.  **数据集类**：在 `dataset.py` 中为您的数据集实现一个新的PyTorch `Dataset` 类（或者如果 `dataset.py` 变得太大，则创建一个适当命名的新模块）。
5.  **加载器函数**：在 `loader.py` 中添加或更新函数，为您的新数据集创建 `DataLoader` 实例。
6.  **更新此README**：记录新数据集子目录和任何特定的组织详细信息。

## 使用说明

此目录中的模块被 `scripts/` 中的训练、评估和处理脚本使用。

*   **准备数据**：
    1.  **获取/放置数据**：确保所需的原始数据集存在。这可能涉及运行使用 `fetcher.py` 的脚本，或手动下载并将数据放入适当的子目录（例如，`data/im2gps3k/images/`，`data/MP16_Pro_filtered.csv`）。
    2.  **预处理**：如果需要任何预处理步骤（由脚本或文档定义），运行它们。
*   **使用数据模块**：
    *   `dataset.py` 中的 `Dataset` 类通常通过提供数据路径和任何必要的处理器（例如，来自模型的图像或文本处理器）来实例化。
    *   然后使用 `loader.py` 中的 `DataLoader` 函数来包装这些 `Dataset` 对象，以便在模型训练中使用。

**示例工作流程（概念）：**

```python
# 在脚本中（例如，scripts/run_g3.py）

from data.dataset import MP16Dataset # 假设MP16Dataset在此定义
from data.loader import create_mp16_dataloader # 假设有一个辅助函数

# 配置（路径、批量大小等）
data_root = "./data/mp16/" # MP16数据的路径
csv_path = "./data/MP16_Pro_filtered.csv"
image_archive_path = "./data/mp-16-images.tar"
batch_size = 64
# vision_processor和text_processor将来自模型

# 1. 实例化数据集
# mp16_dataset = MP16Dataset(root_path=data_root, text_data_path=csv_path, ...)
# (注：实际的MP16Dataset、im2gps3kDataset、yfcc4kDataset之前在utils.g3_utils.py中；
#  此示例假设它们或类似的类将由data/dataset.py定义/管理)

# 2. 创建DataLoader
# train_dataloader = create_mp16_dataloader(mp16_dataset, batch_size=batch_size, ...)

# 3. 在训练循环中使用
# for batch in train_dataloader:
#     images, texts, coordinates = batch
#     # ... 处理批次 ...
```

参考 `scripts/` 目录中的特定脚本，了解如何实例化和使用这些数据模块的具体示例。
