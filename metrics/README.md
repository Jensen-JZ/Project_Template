# Metrics Directory (`metrics/`)

## Purpose

The `metrics/` directory is designated for housing all code related to the quantitative evaluation of models and experiments within this project. This includes, but is not limited to:

*   **Performance Metrics**: Scripts or modules that define and compute standard evaluation metrics (e.g., accuracy, precision, recall, F1-score for classification; Mean Squared Error (MSE), Mean Absolute Error (MAE) for regression; distance-based accuracy for geo-localization).
*   **Custom Evaluation Logic**: Code for specialized evaluation protocols or domain-specific quantitative measures that go beyond standard library implementations.
*   **Loss Functions (Potentially)**: While primary loss functions are often defined within the `solver/` or directly in training scripts, this directory could also contain definitions for complex or custom loss functions if they are elaborate enough to warrant separate modules.
*   **Analysis Scripts**: Scripts that process raw model outputs or logs to generate aggregated metrics or visualizations.

The goal is to centralize evaluation logic, making it reusable, maintainable, and easy to understand.

## Current Structure

Currently, the `metrics/` directory primarily contains this `README.md` file. As the project evolves, any new modules or scripts related to metrics will be added here.

*   **`README.md`**: This file, providing an overview of the `metrics/` directory.

*(If any metric-specific Python modules or scripts (e.g., `accuracy.py`, `geodistance.py`) are added, they will be listed here.)*

## Defining and Using Metrics

Metrics are typically defined as functions or classes within Python modules created in this directory.

**Defining a Metric (Example):**

Let's say you want to define a metric for calculating accuracy at various distance thresholds for a geo-localization task. You could create a file `metrics/geolocation_metrics.py`:

```python
# metrics/geolocation_metrics.py
import numpy as np

def great_circle_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def accuracy_at_thresholds(predicted_coords, true_coords, thresholds_km):
    """
    Calculate the percentage of predictions within given distance thresholds.

    Args:
        predicted_coords (np.array): Array of shape (N, 2) with [lat, lon] predictions.
        true_coords (np.array): Array of shape (N, 2) with [lat, lon] true values.
        thresholds_km (list of float): List of distance thresholds in kilometers.

    Returns:
        dict: A dictionary where keys are thresholds and values are accuracies.
    """
    accuracies = {}
    distances = np.array([
        great_circle_distance(p[0], p[1], t[0], t[1])
        for p, t in zip(predicted_coords, true_coords)
    ])

    for t_km in thresholds_km:
        accuracy = np.mean(distances <= t_km) * 100
        accuracies[f"acc_{t_km}km"] = accuracy
    return accuracies
```

**Using a Metric:**

These defined metrics can then be imported and used within your evaluation scripts or the main solver/training loop.

```python
# In an evaluation script or solver.py

from metrics.geolocation_metrics import accuracy_at_thresholds
import numpy as np

# Assume model_outputs and ground_truth are available
# model_outputs = np.array([[40.7128, -74.0060], ...]) # Predicted [lat, lon]
# ground_truth = np.array([[40.7580, -73.9855], ...])  # True [lat, lon]

thresholds = [1, 25, 200, 750, 2500] # Kilometers
eval_metrics = accuracy_at_thresholds(model_outputs, ground_truth, thresholds)

print("Evaluation Metrics:")
for metric_name, value in eval_metrics.items():
    print(f"{metric_name}: {value:.2f}%")

# These metrics would then be logged or reported.
```

## Adding New Metrics

1.  **Create a New Module (or Augment Existing)**:
    *   If your new metric(s) belong to an existing category (e.g., more geo-localization metrics), add them to the relevant existing file (e.g., `metrics/geolocation_metrics.py`).
    *   If the metric represents a new category, create a new Python file (e.g., `metrics/semantic_similarity.py`).
2.  **Define Your Metric Function/Class**:
    *   Implement the logic for your metric. Ensure functions are well-documented, explaining inputs, outputs, and the calculation performed.
    *   Make sure inputs are clearly defined (e.g., expecting raw model logits, probabilities, predicted class labels, coordinates).
3.  **Ensure Reusability**: Design metrics to be as general as possible, but don't be afraid to make them specific if the task demands it.
4.  **Update this README**:
    *   If you added a new module, list and describe it under the "Current Structure" section.
    *   Optionally, provide a brief example or note under "Defining and Using Metrics" if the new metric introduces a significantly different usage pattern.
5.  **Consider Unit Tests**: For complex metrics, adding unit tests is highly recommended to ensure correctness. Place these in the project's testing directory.
6.  **Integration**: Plan how this metric will be integrated into the training/evaluation pipeline. This might involve modifying `solver.py` or evaluation scripts in `scripts/` to call your new metric function and log its results.

By following these guidelines, the `metrics/` directory will effectively support robust and varied evaluations for the project.
