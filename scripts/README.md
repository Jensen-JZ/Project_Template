# Scripts Directory

## Purpose

The `scripts/` directory is the central location for various executable scripts used in this project. These scripts cover a range of functionalities including model training, evaluation, data processing, index building, and prediction generation.

## Current Structure & Script Descriptions

This directory currently contains the following scripts:

*   **`run_g3.py`**:
    *   **Purpose**: Main training script for the G3 model. It handles data loading, model initialization, the training loop, optimization, and saving checkpoints.
*   **`g3_index_search.py`**:
    *   **Purpose**: Script used for building a FAISS index from model embeddings (likely image embeddings from the G3 model) and then searching this index. This is often used for tasks like candidate retrieval in geo-localization or for geo-diversification and verification steps.
*   **`g3_llm_predict.py`**:
    *   **Purpose**: Generates predictions using a Large Language Model (LLM) in conjunction with the G3 model, likely by taking visual features or initial G3 predictions and refining them or generating textual descriptions/justifications for geolocalization. This version might use a specific LLM API (e.g., OpenAI).
*   **`g3_llm_predict_hf.py`**:
    *   **Purpose**: Similar to `g3_llm_predict.py`, but specifically adapted for use with Hugging Face (HF) compatible LLMs. It handles loading HF models and processors for generating predictions.
*   **`g3_aggregate_llm_predictions.py`**:
    *   **Purpose**: Aggregates and processes the outputs from the LLM prediction scripts (`g3_llm_predict.py`, `g3_llm_predict_hf.py`). This could involve consolidating predictions from multiple runs, different models, or different data splits into a final usable format.
*   **`train.sh`**:
    *   **Purpose**: An example shell script that demonstrates how to execute the training process, likely by calling `run_g3.py` with appropriate command-line arguments and environment configurations.

## Usage

Most Python scripts in this directory are designed to be run from the command line and may accept various arguments to control their behavior.

**General Usage Pattern (for Python scripts):**

```bash
python scripts/<script_name>.py --argument1 value1 --argument2 value2
```

**Specific Examples:**

*   **`run_g3.py`**:
    *   This script is typically run using `accelerate launch` for distributed training or `python` for single-GPU/CPU training.
    *   Common parameters might include batch size, learning rate, number of epochs, paths to data, and checkpoint directories. Refer to the script's internal argument parser (e.g., using `argparse`) for a full list of options.
    *   Example (conceptual):
        ```bash
        accelerate launch scripts/run_g3.py --batch_size 32 --learning_rate 1e-4 --epochs 10 --data_path /path/to/dataset
        ```

*   **`g3_index_search.py`**:
    *   May require arguments specifying the dataset to index, the G3 model checkpoint to use for generating embeddings, and paths for saving/loading the FAISS index.
    *   Example (conceptual):
        ```bash
        python scripts/g3_index_search.py --dataset im2gps3k --index_name g3_im2gps3k_index --checkpoint_path ./checkpoints/g3.pth
        ```

*   **`g3_llm_predict.py` / `g3_llm_predict_hf.py`**:
    *   Will likely require API keys (for `g3_llm_predict.py`), model identifiers, paths to input data (e.g., images or CSV files with image IDs), and output file paths.
    *   Environment variables for API keys might be needed.
    *   Example (conceptual for HF version):
        ```bash
        python scripts/g3_llm_predict_hf.py --model_path ./llava-next-8b-llama3 --input_csv data/im2gps3k/input_data.csv --output_csv data/im2gps3k/llm_predictions.csv
        ```

*   **`train.sh`**:
    *   This is an executable shell script.
    *   Usage: `./scripts/train.sh`
    *   You may need to modify the script internally to set specific parameters or paths according to your environment. Ensure it has execute permissions (`chmod +x scripts/train.sh`).

**Environment Setup:**

*   Ensure all dependencies listed in `requirements.txt` are installed in your Python environment.
*   Some scripts might require specific environment variables to be set (e.g., API keys for LLM services, CUDA_VISIBLE_DEVICES for GPU selection). Check the individual script's documentation or code for such requirements.
*   Ensure that your `PYTHONPATH` is set up correctly so that modules within the project (e.g., from `models/`, `utils/`) can be imported by the scripts. Typically, running scripts from the project's root directory handles this, or you might need to add the root to `PYTHONPATH`:
    ```bash
    export PYTHONPATH=$PYTHONPATH:/path/to/your/project_root
    ```

## Adding New Scripts

1.  **Develop your script**: Create your new Python script (e.g., `my_new_script.py`) or shell script.
2.  **Place it in `scripts/`**: Add the file to this directory.
3.  **Add command-line arguments (if Python)**: Use `argparse` or a similar library to handle command-line parameters for configurability.
4.  **Document your script**: Add comments within your script and, importantly, update this `README.md` file.
    *   Add your script to the list under "Current Structure & Script Descriptions".
    *   Provide a brief description of its purpose.
    *   Add a subsection under "Usage" explaining how to run it, including any key parameters.
5.  **Make shell scripts executable**: If you add a new shell script, ensure it has execute permissions (e.g., `chmod +x scripts/my_new_script.sh`).
6.  **Consider dependencies**: If your script introduces new major dependencies, ensure they are added to `requirements.txt`.
