# Data Processing Directory (`dataprocessing/`)

[English](#english) | [中文](#chinese)

<a name="english"></a>
## Purpose

The `dataprocessing/` directory is dedicated to housing Python modules that handle various aspects of data transformation, aggregation, cleaning, and pre/post-processing pipelines. This includes tasks that prepare data for model consumption, consolidate results from different sources, or transform raw outputs into more structured and usable formats.

The key goals for modules in this directory are:
*   **Data Transformation**: Modifying data structure or content (e.g., parsing, type conversion, feature engineering).
*   **Data Aggregation**: Combining data from multiple files or sources into a unified representation.
*   **Data Cleaning**: Handling missing values, correcting errors, or standardizing data formats.
*   **Pipeline Creation**: Building multi-step processes for systematic data manipulation.

## Current Structure

This directory currently contains the following key module:

*   **`g3_prediction_aggregation.py`**: Contains logic for aggregating and consolidating LLM (Large Language Model) prediction results from various CSV files.

## Module: `g3_prediction_aggregation.py`

*   **Main Role**: The primary function of `g3_prediction_aggregation.py` is to take several CSV files containing different types of LLM predictions (e.g., zero-shot, RAG-based with varying numbers of candidates) and merge them into a single, comprehensive CSV file. This consolidated file typically enriches a raw dataset manifest with the varied LLM predictions.

*   **Functionality Overview**:
    *   **Reading Multiple CSVs**: The module's main function (`aggregate_llm_predictions`) reads a raw data manifest CSV and several prediction CSVs (zero-shot, RAG-5, RAG-10, RAG-15).
    *   **Parsing Prediction Strings**: It parses columns that contain string representations of LLM responses (often lists of JSON strings) to extract specific data points, such as latitude and longitude coordinates. This involves using `ast.literal_eval` for string-to-list/dict conversion and regular expressions (`re.findall`) as a fallback for coordinate extraction.
    *   **Merging Data**: The extracted prediction data is then added as new columns to the raw data manifest. For instance, if an LLM provided 10 coordinate pairs for a zero-shot prediction, this would result in 20 new columns (e.g., `zs_0_latitude`, `zs_0_longitude`, ..., `zs_9_latitude`, `zs_9_longitude`).
    *   **Output**: The final, enriched dataframe is saved to a new CSV file.

*   **Intended Usage**:
    *   The core functionality is encapsulated in the `aggregate_llm_predictions(args)` function.
    *   This function is designed to be imported and called by a launcher script, typically `scripts/execute_g3_aggregation.py`.
    *   The launcher script is responsible for handling command-line arguments (using `argparse`) to specify the paths for all input CSV files (raw data, various prediction files) and the path for the output (aggregated) CSV file. It also passes a parameter for the number of predictions expected per LLM response (e.g., `n_predictions_per_response`).

    **Conceptual Example (from `scripts/execute_g3_aggregation.py`):**
    ```python
    # In scripts/execute_g3_aggregation.py
    from dataprocessing.g3_prediction_aggregation import aggregate_llm_predictions
    # ... (argparse setup for args, including args.raw_df_path, args.zs_pred_path, etc.) ...

    # Call the aggregation function with parsed arguments
    aggregate_llm_predictions(args)
    ```

## Adding New Data Processing Modules

1.  **Define the Scope**: Clearly identify the data processing task (e.g., a new type of data cleaning, a specific feature engineering pipeline, a new aggregation logic).
2.  **Create a New Module**: Add a new Python file to the `dataprocessing/` directory (e.g., `my_data_cleaner.py` or `feature_engineering_pipeline.py`).
3.  **Implement Core Logic**:
    *   Define functions and/or classes within this new module to perform the data processing tasks.
    *   Ensure these components are well-documented with clear docstrings.
    *   Functions should ideally accept an `args` object or specific parameters for configuration (e.g., input/output paths, processing parameters).
4.  **Launcher Script (Recommended)**:
    *   If the new data processing module is intended to be run as a standalone step, create a corresponding launcher script in the `scripts/` directory (e.g., `scripts/execute_my_data_cleaning.py`).
    *   This launcher script should use `argparse` to handle command-line arguments and then import and call the relevant functions/classes from your new module in `dataprocessing/`.
5.  **Documentation**:
    *   Update this `README.md` to include a description of your new module under the "Current Structure" section.
    *   Briefly explain its role and key functionalities.
    *   If applicable, mention its intended usage and the associated launcher script.
6.  **Dependencies**: If your new module relies on external libraries not already in the project, add them to `requirements.txt`.

By organizing data processing logic in this directory, the project ensures that these crucial steps are maintainable, reusable, and clearly separated from other concerns like model definition or training.

<a name="chinese"></a>
# 数据处理目录 (`dataprocessing/`)

## 目的

`dataprocessing/` 目录专门用于存放处理数据转换、聚合、清洗和预/后处理流程的Python模块。这包括为模型消费准备数据、整合不同来源的结果，或将原始输出转换为结构化且可用的格式等任务。

此目录中模块的主要目标是：
*   **数据转换**：修改数据结构或内容（例如，解析、类型转换、特征工程）。
*   **数据聚合**：将多个文件或来源的数据合并为统一表示。
*   **数据清洗**：处理缺失值、纠正错误或标准化数据格式。
*   **流程创建**：构建用于系统性数据处理的多步骤流程。

## 当前结构

此目录当前包含以下关键模块：

*   **`g3_prediction_aggregation.py`**：包含用于聚合和整合来自各种CSV文件的LLM（大型语言模型）预测结果的逻辑。

## 模块：`g3_prediction_aggregation.py`

*   **主要作用**：`g3_prediction_aggregation.py`的主要功能是将包含不同类型LLM预测（例如，零样本、具有不同候选数量的RAG）的多个CSV文件合并为单个综合CSV文件。这个整合文件通常用各种LLM预测丰富原始数据集清单。

*   **功能概述**：
    *   **读取多个CSV**：模块的主要函数（`aggregate_llm_predictions`）读取原始数据清单CSV和几个预测CSV（零样本、RAG-5、RAG-10、RAG-15）。
    *   **解析预测字符串**：它解析包含LLM响应字符串表示（通常是JSON字符串列表）的列，以提取特定数据点，如经纬度坐标。这涉及使用`ast.literal_eval`进行字符串到列表/字典的转换，以及使用正则表达式（`re.findall`）作为坐标提取的备选方法。
    *   **合并数据**：提取的预测数据然后作为新列添加到原始数据清单中。例如，如果LLM为零样本预测提供了10对坐标，这将产生20个新列（例如，`zs_0_latitude`、`zs_0_longitude`、...、`zs_9_latitude`、`zs_9_longitude`）。
    *   **输出**：最终的丰富数据框被保存到新的CSV文件中。

*   **预期用法**：
    *   核心功能封装在`aggregate_llm_predictions(args)`函数中。
    *   该函数设计为由启动脚本导入和调用，通常是`scripts/execute_g3_aggregation.py`。
    *   启动脚本负责处理命令行参数（使用`argparse`）来指定所有输入CSV文件（原始数据、各种预测文件）的路径以及输出（聚合）CSV文件的路径。它还传递一个参数，用于指定每个LLM响应预期的预测数量（例如，`n_predictions_per_response`）。

    **概念示例（来自`scripts/execute_g3_aggregation.py`）：**
    ```python
    # 在 scripts/execute_g3_aggregation.py 中
    from dataprocessing.g3_prediction_aggregation import aggregate_llm_predictions
    # ... (argparse设置参数，包括args.raw_df_path, args.zs_pred_path等) ...

    # 使用解析的参数调用聚合函数
    aggregate_llm_predictions(args)
    ```

## 添加新的数据处理模块

1.  **定义范围**：明确识别数据处理任务（例如，新型数据清洗、特定特征工程流程、新的聚合逻辑）。
2.  **创建新模块**：向`dataprocessing/`目录添加新的Python文件（例如，`my_data_cleaner.py`或`feature_engineering_pipeline.py`）。
3.  **实现核心逻辑**：
    *   在这个新模块中定义执行数据处理任务的函数和/或类。
    *   确保这些组件有良好的文档，包含清晰的文档字符串。
    *   函数应该理想地接受一个`args`对象或特定参数作为配置（例如，输入/输出路径、处理参数）。
4.  **启动脚本（推荐）**：
    *   如果新的数据处理模块旨在作为独立步骤运行，在`scripts/`目录中创建相应的启动脚本（例如，`scripts/execute_my_data_cleaning.py`）。
    *   此启动脚本应使用`argparse`处理命令行参数，然后从`dataprocessing/`中的新模块导入并调用相关函数/类。
5.  **文档**：
    *   更新此`README.md`，在"当前结构"部分包含新模块的描述。
    *   简要解释其角色和关键功能。
    *   如适用，提及其预期用法和相关的启动脚本。
6.  **依赖项**：如果您的新模块依赖于项目中尚未包含的外部库，将它们添加到`requirements.txt`中。

通过在此目录中组织数据处理逻辑，该项目确保了这些关键步骤是可维护、可重用的，并且与其他关注点（如模型定义或训练）明确分离。
