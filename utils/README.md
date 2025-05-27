# Utilities Directory (`utils/`)

## Purpose

The `utils/` directory is a collection of various utility functions and helper modules designed to support different tasks across the project. These modules provide common, reusable code for operations such as checkpoint management, file input/output, image processing, logging, and other miscellaneous tasks, helping to keep the main project scripts cleaner and more focused on their specific logic.

## Current Structure & Module Descriptions

This directory currently contains the following Python modules and subdirectories:

*   **`checkpoint.py`**:
    *   **Description**: Contains utilities for saving and loading model checkpoints during training and for inference. This helps in resuming training or deploying trained models.
*   **`file.py`**:
    *   **Description**: Provides helper functions for various file input/output operations, such as reading from or writing to different file formats, path manipulations, etc.
*   **`g3_utils.py`**:
    *   **Description**: Houses utility functions and classes specifically tailored for the G3 model and its associated datasets. This notably includes `Dataset` classes like `MP16Dataset`, `im2gps3kDataset`, and `yfcc4kDataset` which were originally part of the G3 model's codebase and handle specific data loading and preprocessing for these datasets.
*   **`image.py`**:
    *   **Description**: Includes utilities for image processing tasks, such as transformations, loading, saving, or format conversions that might be commonly needed.
*   **`logger.py`**:
    *   **Description**: Provides functionalities for setting up and managing logging across the project. This ensures consistent logging behavior and formatting.
*   **`misc.py`**:
    *   **Description**: A collection of miscellaneous helper functions that don't fit neatly into other categories but provide useful, general-purpose functionalities.
*   **`model.py`**:
    *   **Description**: Contains general utilities related to model handling, which could include functions for model inspection, parameter counting, or other generic model-related operations not specific to a particular architecture.
*   **`rff/` (subdirectory)**:
    *   **Description**: This directory contains modules related to the implementation of Random Fourier Features (RFF).
    *   `rff/functional.py`: Provides functional implementations of RFF operations, such as sampling and encoding functions.
    *   `rff/layers.py`: Defines PyTorch `nn.Module` layers for RFF, like `GaussianEncoding`, `BasicEncoding`, and `PositionalEncoding`, which can be incorporated into neural network models.

## How to Use Utilities

To use a utility function or class from this directory, simply import it into your Python script.

**Example:**

```python
# In a script located elsewhere in the project (e.g., scripts/run_g3.py)

from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.g3_utils import MP16Dataset # If using datasets defined here
from utils.rff.layers import GaussianEncoding # For model components

# Setup logging
logger = setup_logger(__name__)
logger.info("This is an informational message.")

# Using a dataset class from g3_utils.py
# (Assuming MP16Dataset is now primarily managed or imported via g3_utils.py)
# mp16_data = MP16Dataset(root_path="data/mp16", ...)

# Using a model component from rff/
# rff_encoding_layer = GaussianEncoding(sigma=10.0, input_size=2, encoded_size=256)

# Example of checkpointing (conceptual)
# if should_save_checkpoint:
#     save_checkpoint(model, optimizer, epoch, filepath="my_checkpoint.pth")

# if should_load_checkpoint:
#     model, optimizer, epoch = load_checkpoint(model, optimizer, filepath="my_checkpoint.pth")
```

Ensure that your Python environment is set up correctly (e.g., `PYTHONPATH` includes the project root) so that these modules can be found.

## Adding New Utilities

1.  **Determine Scope**:
    *   If the utility is very specific to a particular model (like G3-specific dataset handling), consider adding it to an existing relevant module (e.g., `g3_utils.py`) or creating a new specific utility file if `g3_utils.py` becomes too large.
    *   If the utility is general (e.g., new file operations, generic math functions), add it to an appropriate existing module (`file.py`, `misc.py`) or create a new module if it represents a distinct category of utilities (e.g., `my_new_category_utils.py`).
    *   If creating a new category that involves multiple related files (like `rff/`), create a new subdirectory.
2.  **Implement the Utility**: Write your function or class. Ensure it is well-documented with clear docstrings explaining its purpose, arguments, and return values.
3.  **Add Imports**: If your new utility depends on other packages, ensure these are standard libraries or are listed in the project's `requirements.txt`.
4.  **Update this README**:
    *   If you added a new file or subdirectory, list and describe it under the "Current Structure & Module Descriptions" section.
    *   If you added a significant new function to an existing module that users should be aware of, you might briefly mention its capability in the module's description.
5.  **Consider Unit Tests**: For complex or critical utilities, adding unit tests in the appropriate testing directory is highly recommended.

By following these guidelines, the `utils/` directory can remain organized and provide a robust set of helper tools for the project.
