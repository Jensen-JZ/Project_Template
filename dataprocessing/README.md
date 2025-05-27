# Data Processing Directory (`dataprocessing/`)

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
