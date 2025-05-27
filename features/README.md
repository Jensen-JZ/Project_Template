- [English](#features-directory-features)
- [中文](#特性目录-features)

# Features Directory (`features/`)

## Purpose

The `features/` directory is designed to house Python modules that perform complex operations related to feature extraction, model-specific operational logic beyond simple forward passes, or intricate data transformations that result in derived 'features' or structured outputs. This can include:

*   **Specialized Model Operations**: Functions that utilize trained models for specific tasks like generating embeddings for indexing, performing nearest neighbor searches, or complex evaluation routines.
*   **Feature Engineering**: Modules that transform raw or processed data into feature representations suitable for model consumption or for specific analytical tasks.
*   **Domain-Specific Logic**: Encapsulation of logic that is particular to a certain domain or model capability, such as geospatial calculations, advanced image analysis routines, or natural language processing feature extraction.

The aim is to separate these more involved, often multi-step, operations from the core model definitions (`models/`) and the basic training loops (`training/`).

## Current Structure

This directory currently contains the following key module:

*   **`g3_geospatial_operations.py`**: Provides functionalities for G3 model-based geospatial operations, including FAISS indexing, searching, and evaluation.

## Module: `g3_geospatial_operations.py`

*   **Main Role**: This module centralizes the logic for using the G3 model to perform geospatial tasks. It leverages the G3 model's embeddings to build and search a FAISS index, and then evaluates the quality of these search results, often involving a re-ranking step using LLM-generated candidate coordinates.

*   **Key Components**:
    *   **`GeoImageDataset` Class**: A PyTorch `Dataset` class specifically designed for the evaluation phase. It loads query images and their associated candidate GPS coordinates (derived from LLM predictions and initial search results) for re-ranking by the G3 model's location understanding capabilities.
    *   **`build_index(args)` Function**: Takes arguments (typically including model path, device, and index configuration) and uses the G3 model to generate embeddings for a dataset (e.g., MP16). It then builds a FAISS index from these embeddings and saves it to disk.
    *   **`search_index(args, index, topk)` Function**: Takes arguments, a loaded FAISS index, and a `topk` value. It loads a query dataset (e.g., im2gps3k, yfcc4k), generates G3 embeddings for these queries, and searches the provided FAISS index to find the top `k` nearest neighbors.
    *   **`evaluate(args, I)` Function**: Takes arguments and the search results (indices `I`). It loads the query dataset's ground truth, the database's ground truth (e.g., MP16), and potentially LLM-generated candidate coordinates. It then performs an evaluation, which can include:
        1.  Initial evaluation based on top-1 FAISS search result.
        2.  Re-ranking of candidates using the `GeoImageDataset` and G3 model's image and location encoders to select the best coordinate set.
        3.  Calculation of geodesic distances and accuracy metrics at various thresholds.

*   **Intended Usage**:
    *   The functions within `g3_geospatial_operations.py` are designed to be imported and orchestrated by a launcher script.
    *   The primary launcher for this module is `scripts/execute_g3_geo_ops.py`. This script handles argument parsing (model paths, dataset paths, FAISS index paths, etc.) and then calls `build_index`, `search_index`, and `evaluate` in the appropriate sequence, managing FAISS index loading/GPU transfer as needed.

    **Conceptual Example (from `scripts/execute_g3_geo_ops.py`):**
    ```python
    # In scripts/execute_g3_geo_ops.py
    from features.g3_geospatial_operations import build_index, search_index, evaluate, res as faiss_gpu_resources
    # ... (argparse setup for args) ...

    # Build index if it doesn't exist
    if not os.path.exists(index_file_path):
        build_index(args)

    # Load index and search
    faiss_index_cpu = faiss.read_index(index_file_path)
    # ... (optional: move index to GPU using faiss_gpu_resources) ...
    D_results, I_results = search_index(args, faiss_index_gpu_or_cpu, args.search_topk)

    # Evaluate results
    evaluate(args, I_results)
    ```

## Adding New Feature-Related Modules

1.  **Define Scope**: Identify the set of operations or feature engineering tasks you want to encapsulate. This could be related to a new model, a new type of data, or a new analytical method.
2.  **Create a New Module**: Add a new Python file to the `features/` directory (e.g., `my_new_feature_extractor.py`).
3.  **Implement Core Logic**: Define functions and/or classes within this new module. Ensure they are well-documented.
    *   Functions should ideally accept an `args` object or specific parameters for configuration (e.g., model paths, input/output data paths, processing parameters).
4.  **Launcher Script**: Create a corresponding launcher script in the `scripts/` directory (e.g., `scripts/execute_my_feature_extraction.py`). This script will:
    *   Use `argparse` to handle command-line arguments.
    *   Import and call the functions/classes from your new module in `features/`.
5.  **Documentation**:
    *   Update this `README.md` to include a description of your new module under the "Current Structure" section.
    *   Briefly explain its role and key components.
    *   Mention its intended usage and the associated launcher script.

By organizing complex operations and feature engineering tasks in this directory, the project maintains a clear separation of concerns and promotes reusability.

# 特性目录 (`features/`)

## 目的

`features/` 目录旨在存放执行复杂操作的Python模块，这些操作涉及特征提取、超出简单前向传递的模型特定操作逻辑，或产生派生"特征"或结构化输出的复杂数据转换。这包括：

*   **专业模型操作**：利用训练好的模型执行特定任务的函数，如生成用于索引的嵌入、执行最近邻搜索或复杂评估程序。
*   **特征工程**：将原始或处理过的数据转换为适合模型消费或特定分析任务的特征表示的模块。
*   **领域特定逻辑**：封装特定于某个领域或模型能力的逻辑，如地理空间计算、高级图像分析程序或自然语言处理特征提取。

目的是将这些更复杂、通常是多步骤的操作与核心模型定义（`models/`）和基本训练循环（`training/`）分开。

## 当前结构

该目录目前包含以下关键模块：

*   **`g3_geospatial_operations.py`**：提供基于G3模型的地理空间操作功能，包括FAISS索引、搜索和评估。

## 模块：`g3_geospatial_operations.py`

*   **主要角色**：此模块集中了使用G3模型执行地理空间任务的逻辑。它利用G3模型的嵌入来构建和搜索FAISS索引，然后评估这些搜索结果的质量，通常涉及使用LLM生成的候选坐标进行重新排序。

*   **关键组件**：
    *   **`GeoImageDataset` 类**：专为评估阶段设计的PyTorch `Dataset` 类。它加载查询图像及其相关的候选GPS坐标（源自LLM预测和初始搜索结果），以便由G3模型的位置理解能力进行重新排序。
    *   **`build_index(args)` 函数**：接受参数（通常包括模型路径、设备和索引配置）并使用G3模型为数据集（例如MP16）生成嵌入。然后它从这些嵌入构建FAISS索引并将其保存到磁盘。
    *   **`search_index(args, index, topk)` 函数**：接受参数、已加载的FAISS索引和`topk`值。它加载查询数据集（例如im2gps3k、yfcc4k），为这些查询生成G3嵌入，并搜索提供的FAISS索引以找到前`k`个最近邻。
    *   **`evaluate(args, I)` 函数**：接受参数和搜索结果（索引`I`）。它加载查询数据集的真实值、数据库的真实值（例如MP16）以及可能由LLM生成的候选坐标。然后它执行评估，可能包括：
        1.  基于FAISS搜索结果的前1名进行初步评估。
        2.  使用`GeoImageDataset`和G3模型的图像和位置编码器对候选项进行重新排序，以选择最佳坐标集。
        3.  计算各种阈值下的测地距离和准确度指标。

*   **预期用法**：
    *   `g3_geospatial_operations.py`中的函数设计为由启动器脚本导入和编排。
    *   此模块的主要启动器是`scripts/execute_g3_geo_ops.py`。该脚本处理参数解析（模型路径、数据集路径、FAISS索引路径等），然后按适当顺序调用`build_index`、`search_index`和`evaluate`，管理FAISS索引加载/GPU传输。

    **概念示例（来自`scripts/execute_g3_geo_ops.py`）：**
    ```python
    # In scripts/execute_g3_geo_ops.py
    from features.g3_geospatial_operations import build_index, search_index, evaluate, res as faiss_gpu_resources
    # ... (argparse setup for args) ...

    # Build index if it doesn't exist
    if not os.path.exists(index_file_path):
        build_index(args)

    # Load index and search
    faiss_index_cpu = faiss.read_index(index_file_path)
    # ... (optional: move index to GPU using faiss_gpu_resources) ...
    D_results, I_results = search_index(args, faiss_index_gpu_or_cpu, args.search_topk)

    # Evaluate results
    evaluate(args, I_results)
    ```

## 添加新的特性相关模块

1.  **定义范围**：确定你想要封装的操作或特征工程任务集。这可能与新模型、新数据类型或新分析方法相关。
2.  **创建新模块**：向`features/`目录添加一个新的Python文件（例如`my_new_feature_extractor.py`）。
3.  **实现核心逻辑**：在新模块中定义函数和/或类。确保它们有良好的文档记录。
    *   函数理想情况下应该接受一个`args`对象或特定参数进行配置（例如模型路径、输入/输出数据路径、处理参数）。
4.  **启动脚本**：在`scripts/`目录中创建相应的启动脚本（例如`scripts/execute_my_feature_extraction.py`）。此脚本将：
    *   使用`argparse`处理命令行参数。
    *   从你在`features/`中的新模块导入并调用函数/类。
5.  **文档**：
    *   更新此`README.md`，在"当前结构"部分包含你的新模块描述。
    *   简要解释其角色和关键组件。
    *   提及其预期用法和相关的启动脚本。

通过在此目录中组织复杂操作和特征工程任务，项目保持了关注点的清晰分离并促进了重用性。
