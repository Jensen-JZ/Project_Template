# Data Directory

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
