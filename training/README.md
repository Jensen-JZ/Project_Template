# Training Directory (`training/`)

[English](#training-directory-training) | [中文](#训练目录-training)

## Training Directory (`training/`)

## Purpose

The `training/` directory is dedicated to housing Python modules and classes that define and manage the processes for training machine learning models within this project. The primary goal is to encapsulate the complexities of the training pipeline, including model setup, data loading orchestration, optimization procedures, epoch-based training loops, and checkpoint management, into reusable and well-structured components.

## Current Structure

This directory currently contains the following key module:

*   **`g3_trainer.py`**: Defines the `G3Trainer` class, which is responsible for training the G3 geolocalization model.

## Module: `g3_trainer.py`

### `G3Trainer` Class

*   **Model Trained**: The `G3Trainer` class is specifically designed to train the **G3 model** (defined in `models/G3.py`). The G3 model is a neural network architecture tailored for geolocalization tasks, typically involving image, text, and location encoders.

*   **Key Responsibilities**:
    *   **Model Setup**: Initializes the G3 model, including loading any pre-trained components like the `location_encoder.pth` if available. It uses `accelerate` for handling device placement and distributed training preparations.
    *   **Data Loading**: Sets up the `MP16Dataset` (from `utils.g3_utils.py`) and the PyTorch `DataLoader` for efficient batching and iteration over the training data. It utilizes the model's internal vision and text processors for dataset preparation.
    *   **Optimization**: Configures the AdamW optimizer and a StepLR learning rate scheduler.
    *   **Epoch-based Training Loop**: Implements the main training loop (`train()` method) which iterates over a specified number of epochs. Within each epoch, the `_train_epoch()` method processes batches of data, performs forward and backward passes, updates model parameters, and logs progress.
    *   **Checkpointing**: Saves model checkpoints (the unwrapped model state) at the end of each epoch to allow for resumption of training or later use of the trained model. This is handled only on the main process in a distributed setup.

*   **Intended Usage**:
    *   The `G3Trainer` class is not typically run directly. Instead, it is designed to be instantiated and used by a launcher script located in the `scripts/` directory.
    *   The launcher script (e.g., `scripts/execute_g3_training.py`) is responsible for parsing command-line arguments (for configurations like batch size, learning rate, number of epochs, etc., though current `G3Trainer` has defaults for many) and then creating an instance of `G3Trainer`.
    *   After instantiation, the launcher script calls the `trainer.train(num_epochs=...)` method to start the training process.

    **Conceptual Example (from a launcher script):**
    ```python
    # In scripts/execute_g3_training.py
    from training.g3_trainer import G3Trainer

    # ... (argument parsing) ...

    trainer = G3Trainer(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        # ... other parameters ...
    )
    trainer.train(num_epochs=args.epochs)
    ```

## Adding New Trainers or Training Modules

When new models or significantly different training procedures are introduced, you can extend this directory:

1.  **Create a New Trainer Class**:
    *   If training a new model (e.g., "MyNewModel"), create a new Python file (e.g., `my_new_model_trainer.py`) in this `training/` directory.
    *   Define a new trainer class (e.g., `MyNewModelTrainer`) within this file.
    *   This class should encapsulate all logic specific to training "MyNewModel," similar to how `G3Trainer` handles G3.
2.  **Abstract Common Logic (Optional)**:
    *   If multiple trainers share a lot of common functionality (e.g., basic Accelerator setup, optimizer creation, checkpointing logic), consider creating a `BaseTrainer` class within a new `base_trainer.py` module in this directory. New trainers can then inherit from `BaseTrainer` to reduce code duplication.
3.  **Supporting Modules**:
    *   If a new training process requires specialized utility functions (e.g., unique data sampling strategies specific to a trainer, complex logging during training), these can be included in the new trainer's module or, if more general, in a new utility module within `training/` (e.g., `training_utils.py`).
4.  **Launcher Script**:
    *   Create a corresponding launcher script in the `scripts/` directory (e.g., `scripts/execute_my_new_model_training.py`) that instantiates and calls your new trainer.
5.  **Documentation**:
    *   Update this `README.md` to include a description of your new trainer module and class.
    *   Document the new trainer class and its methods thoroughly with docstrings.

By following this structure, the training logic for different models and experiments remains organized and maintainable.

## 训练目录 (`training/`)

## 目的

`training/` 目录专门用于存放定义和管理本项目中机器学习模型训练流程的Python模块和类。主要目标是将训练流水线的复杂性（包括模型设置、数据加载编排、优化过程、基于epoch的训练循环和检查点管理）封装到可重用且结构良好的组件中。

## 当前结构

这个目录目前包含以下关键模块：

*   **`g3_trainer.py`**：定义了负责训练G3地理定位模型的`G3Trainer`类。

## 模块：`g3_trainer.py`

### `G3Trainer` 类

*   **训练的模型**：`G3Trainer`类专门设计用于训练**G3模型**（在`models/G3.py`中定义）。G3模型是一种为地理定位任务定制的神经网络架构，通常涉及图像、文本和位置编码器。

*   **主要职责**：
    *   **模型设置**：初始化G3模型，包括加载任何预训练组件，如有可用的`location_encoder.pth`。它使用`accelerate`来处理设备放置和分布式训练准备。
    *   **数据加载**：设置`MP16Dataset`（来自`utils.g3_utils.py`）和PyTorch的`DataLoader`，用于高效批处理和迭代训练数据。它使用模型的内部视觉和文本处理器进行数据集准备。
    *   **优化**：配置AdamW优化器和StepLR学习率调度器。
    *   **基于Epoch的训练循环**：实现主训练循环（`train()`方法），该方法迭代指定数量的epoch。在每个epoch中，`_train_epoch()`方法处理数据批次，执行前向和后向传播，更新模型参数，并记录进度。
    *   **检查点保存**：在每个epoch结束时保存模型检查点（解包装的模型状态），以允许恢复训练或以后使用训练好的模型。在分布式设置中，这仅在主进程上处理。

*   **预期用途**：
    *   `G3Trainer`类通常不直接运行。相反，它被设计为由位于`scripts/`目录中的启动器脚本实例化和使用。
    *   启动器脚本（例如，`scripts/execute_g3_training.py`）负责解析命令行参数（用于批量大小、学习率、epoch数等配置，尽管当前的`G3Trainer`对许多参数有默认值）然后创建`G3Trainer`的实例。
    *   实例化后，启动器脚本调用`trainer.train(num_epochs=...)`方法开始训练过程。

    **概念示例（来自启动器脚本）：**
    ```python
    # 在 scripts/execute_g3_training.py 中
    from training.g3_trainer import G3Trainer

    # ... (参数解析) ...

    trainer = G3Trainer(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        # ... 其他参数 ...
    )
    trainer.train(num_epochs=args.epochs)
    ```

## 添加新的训练器或训练模块

当引入新模型或显著不同的训练程序时，你可以扩展此目录：

1.  **创建新的训练器类**：
    *   如果训练新模型（例如，"MyNewModel"），在此`training/`目录中创建一个新的Python文件（例如，`my_new_model_trainer.py`）。
    *   在这个文件中定义一个新的训练器类（例如，`MyNewModelTrainer`）。
    *   这个类应该封装所有特定于训练"MyNewModel"的逻辑，类似于`G3Trainer`如何处理G3。
2.  **抽象公共逻辑（可选）**：
    *   如果多个训练器共享大量共同功能（例如，基本的Accelerator设置、优化器创建、检查点逻辑），考虑在此目录中的新`base_trainer.py`模块中创建一个`BaseTrainer`类。新的训练器然后可以继承自`BaseTrainer`以减少代码重复。
3.  **支持模块**：
    *   如果新的训练过程需要专门的工具函数（例如，特定于训练器的唯一数据采样策略、训练期间的复杂记录），这些可以包含在新训练器的模块中，或者，如果更一般，则放在`training/`中的新工具模块中（例如，`training_utils.py`）。
4.  **启动器脚本**：
    *   在`scripts/`目录中创建相应的启动器脚本（例如，`scripts/execute_my_new_model_training.py`），该脚本实例化并调用你的新训练器。
5.  **文档**：
    *   更新此`README.md`以包含对你的新训练器模块和类的描述。
    *   使用文档字符串彻底记录新训练器类及其方法。

通过遵循这种结构，不同模型和实验的训练逻辑保持有组织和可维护性。
