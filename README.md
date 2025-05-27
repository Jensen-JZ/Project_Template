# Project Title: Advanced Geolocation Model Framework

## Overview

This project provides a comprehensive framework for developing, training, and evaluating advanced geolocation models, with a primary focus on the G3 model architecture. It includes modules for data handling, model definitions, training pipelines, feature extraction, geospatial operations, LLM-based prediction services, and results aggregation. The structure is designed to be modular, allowing for easy extension and experimentation with new models, data sources, and evaluation techniques.

## Directory Structure

The project is organized into the following key directories:

*   **`models/`**: Contains definitions for core neural network architectures.
    *   `G3.py`: Defines the G3 geolocalization model.
    *   `README.md`: Details on models and how to add new ones.
*   **`data/`**: Manages datasets and data loading utilities.
    *   `dataset.py`, `loader.py`, `fetcher.py`: Modules for dataset definition, loading, and fetching.
    *   `README.md`: Explains data organization and module usage.
*   **`dataprocessing/`**: Houses modules for data transformation, aggregation, and (post-)processing.
    *   `g3_prediction_aggregation.py`: Aggregates LLM prediction results.
    *   `README.md`: Details on data processing modules.
*   **`features/`**: For modules performing complex feature extraction or specialized model operations.
    *   `g3_geospatial_operations.py`: Handles FAISS indexing, searching, and evaluation using G3 model embeddings.
    *   `README.md`: Information on feature-related modules.
*   **`services/`**: Encapsulates complex functionalities, especially those interacting with external APIs or managing distinct operational logic.
    *   `llm_geolocation.py`: Provides classes for geolocation using LLMs (OpenAI and local Hugging Face models).
    *   `README.md`: Describes available services.
*   **`training/`**: Contains modules for defining and managing model training processes.
    *   `g3_trainer.py`: Defines the `G3Trainer` class for training the G3 model.
    *   `README.md`: Explains training components and how to add new trainers.
*   **`solver/`**: Manages core training logic, including loss functions and optimization procedures. (Note: Much of this might be integrated into `training/g3_trainer.py` now).
    *   `loss.py`, `solver.py`, `misc.py`, `utils.py`: Components for the training loop.
    *   `README.md`: Details on solver components.
*   **`scripts/`**: Contains launcher scripts for various workflows (training, evaluation, data processing, etc.).
    *   `execute_*.py`: Launcher scripts for the refactored modules.
    *   `train.sh`: Example shell script for training.
    *   `README.md`: Explains how to use the launcher scripts.
*   **`utils/`**: A collection of general utility functions and helper modules.
    *   Includes modules for checkpointing, file operations, image processing, logging, etc.
    *   `rff/`: Subdirectory for Random Fourier Features implementation.
    *   `README.md`: Describes available utilities.
*   **`metrics/`**: For defining and computing evaluation metrics.
    *   `README.md`: Guidelines on adding and using metrics.
*   **`examples/`**: Contains example scripts demonstrating usage of core components.
    *   `run_g3_example.py`: Shows how to run the G3 model with dummy inputs.
*   **`archive/`**: (As seen in `ls` output) Likely for storing older or deprecated project components.
*   `config.py`: Defines and manages configuration options (via `argparse`) for the generic template framework used by `main.py`. **Note**: While some general settings might be adaptable, G3-specific operations launched via `scripts/` typically manage their own configurations directly. Refer to individual scripts in `scripts/` for their specific arguments.
*   `main.py`: Entry point for the original generic PyTorch template. It uses `config.py` and `solver/solver.py` for a general-purpose training/evaluation loop. **Note**: For G3-specific model operations (training, indexing, etc.), please use the launcher scripts in the `scripts/` directory.
*   `requirements.txt`: Lists project dependencies.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a Python environment:**
    It's recommended to use a virtual environment (e.g., venv, conda).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up PYTHONPATH (if needed):**
    Most scripts are structured to run correctly from the project root. If you encounter import errors, ensure the project root directory is in your `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root
    ```

## Usage

This project uses a modular structure where core logic is implemented in directories like `training/`, `features/`, `services/`, and `dataprocessing/`. These functionalities are typically invoked via launcher scripts located in the `scripts/` directory.

1.  **Explore Launcher Scripts**: Navigate to the `scripts/` directory. The `scripts/README.md` provides an overview of available launcher scripts and their purposes.
2.  **Get Help**: Each Python launcher script supports a `--help` argument that lists its specific command-line options:
    ```bash
    python scripts/<launcher_script_name>.py --help
    ```
    For example:
    ```bash
    python scripts/execute_g3_training.py --help
    python scripts/execute_g3_geo_ops.py --help
    python scripts/execute_openai_geoloc.py --help
    ```
3.  **Running Workflows**:
    *   **Training**: Use `scripts/execute_g3_training.py` to train the G3 model. Configure parameters as needed via command-line arguments. The `scripts/train.sh` script provides an example of how to run this.
    *   **Geospatial Operations (Indexing, Searching, Evaluation)**: Use `scripts/execute_g3_geo_ops.py`.
    *   **LLM Geolocation (OpenAI)**: Use `scripts/execute_openai_geoloc.py`. Requires API keys and appropriate configuration.
    *   **LLM Geolocation (Hugging Face LLaVA)**: Use `scripts/execute_hf_llava_geoloc.py`. Requires a local LLaVA model.
    *   **Prediction Aggregation**: Use `scripts/execute_g3_aggregation.py` to combine various prediction outputs.

Refer to the `README.md` files within each specific directory (`models/`, `data/`, `training/`, etc.) for more detailed information on their respective components and how to extend them.

## Adding New Components

The project is designed for modularity. When adding new functionalities:

*   **New Models**: Add to `models/` and update `models/README.md`. Consider if `models/build.py` needs changes.
*   **New Training Logic**: Create new trainers in `training/` and update `training/README.md`. Add a corresponding launcher in `scripts/`.
*   **New Feature/Geospatial Operations**: Add to `features/` and update `features/README.md`. Add a launcher in `scripts/`.
*   **New Services (e.g., API clients, complex pipelines)**: Add to `services/` and update `services/README.md`. Add a launcher in `scripts/`.
*   **New Data Processing Steps**: Add to `dataprocessing/` and update `dataprocessing/README.md`. Add a launcher in `scripts/`.
*   **New Datasets/Loaders**: Add definitions to `data/dataset.py` or `data/loader.py` and update `data/README.md`.
*   **General Utilities**: Add to appropriate modules in `utils/` and update `utils/README.md`.

Always create or update relevant README files when adding new components or directories.

## Contributing

(Placeholder for contribution guidelines)

## License

(Placeholder for license information - e.g., MIT, Apache 2.0)
