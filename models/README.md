# Models Directory

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
