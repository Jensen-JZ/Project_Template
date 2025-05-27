# Scripts Directory

## Purpose

The `scripts/` directory serves as the primary location for launching various workflows and operations within this project. Most Python scripts in this directory are **launchers** that execute core logic implemented in other dedicated modules (e.g., `training/`, `features/`, `services/`, `dataprocessing/`). This separation helps in organizing the codebase and making the core functionalities reusable.

## Current Structure & Launcher Script Descriptions

This directory currently contains the following launcher scripts:

*   **`execute_g3_training.py`**:
    *   **Purpose**: Launches the training process for the G3 model.
    *   **Core Logic**: Implemented in the `G3Trainer` class within `training/g3_trainer.py`. This script handles parsing training configurations and initiating the trainer.
*   **`execute_g3_geo_ops.py`**:
    *   **Purpose**: Executes various geospatial operations related to the G3 model, such as building FAISS indices from model embeddings, searching these indices, and evaluating search results.
    *   **Core Logic**: Implemented in modules within `features/g3_geospatial_operations.py`.
*   **`execute_openai_geoloc.py`**:
    *   **Purpose**: Runs geolocation prediction tasks using Large Language Models (LLMs) via the OpenAI API. This includes zero-shot predictions and RAG-based predictions.
    *   **Core Logic**: Implemented in the `OpenAIGeolocator` class within `services/llm_geolocation.py`.
*   **`execute_hf_llava_geoloc.py`**:
    *   **Purpose**: Runs geolocation prediction tasks using local Hugging Face LLaVA (or similar compatible) models. Supports zero-shot and RAG-based predictions.
    *   **Core Logic**: Implemented in the `HuggingFaceLlavaGeolocator` class within `services/llm_geolocation.py`.
*   **`execute_g3_aggregation.py`**:
    *   **Purpose**: Aggregates and consolidates LLM prediction results from various sources or runs into a final structured format.
    *   **Core Logic**: Implemented in the `aggregate_llm_predictions` function within `dataprocessing/g3_prediction_aggregation.py`.
*   **`train.sh`**:
    *   **Purpose**: An example shell script that demonstrates how to execute a training process. It likely calls one of the Python launcher scripts (e.g., `execute_g3_training.py`) with specific command-line arguments and environment configurations. This script can be adapted for specific training setups or batch job submissions.

## Usage

All Python launcher scripts in this directory are designed to be run from the command line. They accept various arguments to control their behavior, such as paths to data, model configurations, API keys, etc.

**General Usage Pattern:**

To understand the specific arguments and options available for each launcher script, use the `--help` flag:

```bash
python scripts/<launcher_script_name>.py --help
```
For example:
```bash
python scripts/execute_g3_training.py --help
python scripts/execute_openai_geoloc.py --help
```

**Executing Scripts:**

Once you know the required arguments, you can run the scripts as follows:

```bash
python scripts/<launcher_script_name>.py --argument1 value1 --argument2 value2 ...
```

*   **`train.sh`**:
    *   This is an executable shell script.
    *   Usage: `./scripts/train.sh`
    *   You may need to modify this script internally to set specific parameters or paths accordingto your environment and which Python launcher it invokes. Ensure it has execute permissions (`chmod +x scripts/train.sh`).

**Environment Setup:**

*   Ensure all dependencies listed in `requirements.txt` are installed in your Python environment.
*   Some scripts might require specific environment variables to be set (e.g., API keys for LLM services, `CUDA_VISIBLE_DEVICES`). Refer to the help output of each script or the documentation of the underlying core modules.
*   Ensure that your `PYTHONPATH` is set up correctly so that modules within the project (e.g., from `models/`, `utils/`, `training/`, `features/`, `services/`, `dataprocessing/`) can be imported by the scripts. Typically, running scripts from the project's root directory handles this, or you might need to add the project root to `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root
    ```

## Adding New Launcher Scripts

1.  **Develop Core Logic First**: Implement the primary functionality in a dedicated module (e.g., in `services/`, `features/`, `dataprocessing/`, or a new directory).
2.  **Create the Launcher Script**: Add a new Python script in this `scripts/` directory.
3.  **Implement Argument Parsing**: Use `argparse` to define command-line arguments that your launcher script will accept. These arguments should configure and control the core logic.
4.  **Call Core Logic**: In the launcher script, import the necessary functions or classes from your core module and execute them using the parsed arguments.
5.  **Document the Launcher**: Update this `README.md` file:
    *   Add your new launcher script to the list under "Current Structure & Launcher Script Descriptions".
    *   Provide a brief description of its purpose and point to the module where the core logic resides.
    *   Mention any critical or new usage patterns.
6.  **Make Shell Scripts Executable**: If you add a new shell script, grant it execute permissions (e.g., `chmod +x scripts/my_new_launcher.sh`).
