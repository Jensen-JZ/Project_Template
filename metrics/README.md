# Metrics Directory (`metrics/`)

[English](#english) | [中文](#chinese)

<a name="english"></a>
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

<a name="chinese"></a>
# 指标目录 (`metrics/`)

## 目的

`metrics/` 目录用于存放与本项目中模型和实验的定量评估相关的所有代码。这包括但不限于：

*   **性能指标**：定义和计算标准评估指标的脚本或模块（例如，分类任务中的准确率、精确率、召回率、F1分数；回归任务中的均方误差(MSE)、平均绝对误差(MAE)；地理定位中的基于距离的准确率）。
*   **自定义评估逻辑**：用于专门评估协议或超出标准库实现的特定领域定量测量的代码。
*   **损失函数（可能的）**：虽然主要损失函数通常在 `solver/` 中或直接在训练脚本中定义，但如果它们足够复杂或自定义，需要单独的模块，这个目录也可以包含这些定义。
*   **分析脚本**：处理原始模型输出或日志以生成聚合指标或可视化的脚本。

目标是集中评估逻辑，使其可重用、易于维护和理解。

## 当前结构

目前，`metrics/` 目录主要包含这个 `README.md` 文件。随着项目的发展，与指标相关的任何新模块或脚本将添加到这里。

*   **`README.md`**：此文件，提供 `metrics/` 目录的概述。

*（如果添加了任何特定于指标的Python模块或脚本（例如，`accuracy.py`，`geodistance.py`），它们将在此列出。）*

## 定义和使用指标

指标通常作为函数或类在此目录中创建的Python模块内定义。

**定义指标（示例）：**

假设你想为地理定位任务定义一个在不同距离阈值下计算准确率的指标。你可以创建一个文件 `metrics/geolocation_metrics.py`：

```python
# metrics/geolocation_metrics.py
import numpy as np

def great_circle_distance(lat1, lon1, lat2, lon2):
    """
    计算地球上两点之间的大圆距离
    （以十进制度数指定）。
    """
    # 将十进制度数转换为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # 地球半径（公里）。使用3956表示英里
    return c * r

def accuracy_at_thresholds(predicted_coords, true_coords, thresholds_km):
    """
    计算在给定距离阈值内的预测百分比。

    参数：
        predicted_coords (np.array)：形状为(N, 2)的数组，包含[纬度, 经度]预测值。
        true_coords (np.array)：形状为(N, 2)的数组，包含[纬度, 经度]真实值。
        thresholds_km (float列表)：距离阈值列表，单位为公里。

    返回：
        dict：一个字典，其中键为阈值，值为准确率。
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

**使用指标：**

然后可以在评估脚本或主求解器/训练循环中导入和使用这些已定义的指标。

```python
# 在评估脚本或solver.py中

from metrics.geolocation_metrics import accuracy_at_thresholds
import numpy as np

# 假设model_outputs和ground_truth已可用
# model_outputs = np.array([[40.7128, -74.0060], ...]) # 预测的[纬度, 经度]
# ground_truth = np.array([[40.7580, -73.9855], ...])  # 真实的[纬度, 经度]

thresholds = [1, 25, 200, 750, 2500] # 公里
eval_metrics = accuracy_at_thresholds(model_outputs, ground_truth, thresholds)

print("评估指标：")
for metric_name, value in eval_metrics.items():
    print(f"{metric_name}: {value:.2f}%")

# 这些指标随后将被记录或报告。
```

## 添加新指标

1.  **创建新模块（或增强现有模块）**：
    *   如果你的新指标属于现有类别（例如，更多地理定位指标），将它们添加到相关的现有文件（例如，`metrics/geolocation_metrics.py`）。
    *   如果该指标代表一个新类别，创建一个新的Python文件（例如，`metrics/semantic_similarity.py`）。
2.  **定义你的指标函数/类**：
    *   实现你的指标逻辑。确保函数有良好的文档，解释输入、输出和执行的计算。
    *   确保输入明确定义（例如，期望原始模型logits、概率、预测的类别标签、坐标）。
3.  **确保可重用性**：设计指标尽可能通用，但如果任务需要，不要害怕使其特定化。
4.  **更新此README**：
    *   如果你添加了新模块，在"当前结构"部分列出并描述它。
    *   如果新指标引入了明显不同的使用模式，可以选择在"定义和使用指标"下提供简短示例或说明。
5.  **考虑单元测试**：对于复杂指标，强烈建议添加单元测试以确保正确性。将这些测试放在项目的测试目录中。
6.  **集成**：计划如何将此指标集成到训练/评估管道中。这可能涉及修改`solver.py`或`scripts/`中的评估脚本，以调用你的新指标函数并记录其结果。

通过遵循这些指导原则，`metrics/` 目录将有效地支持项目的健壮和多样化评估。
