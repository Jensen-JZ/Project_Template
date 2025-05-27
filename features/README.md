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
