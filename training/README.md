# Training Directory (`training/`)

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
