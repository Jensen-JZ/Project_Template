# Services Directory (`services/`)

## Purpose

The `services/` directory is designed to house Python modules that encapsulate complex, self-contained functionalities, especially those that interact with external APIs or manage significant, distinct operational logic that acts like a "service." This includes, but is not limited to:

*   **External API Clients**: Modules that provide an interface to third-party services (e.g., cloud-based AI models, data providers).
*   **Complex Processing Units**: Components that manage intricate workflows involving multiple steps, model interactions, and data transformations, such as sophisticated prediction pipelines.
*   **Self-Contained Functionality**: Modules that can be used across different parts of the project or even potentially be decoupled for standalone use, offering a clear API to their capabilities.

The primary goal is to abstract away the internal complexity of these services, providing a clean and easy-to-use interface for other parts of the project, particularly launcher scripts in the `scripts/` directory.

## Current Structure

This directory currently contains the following key module:

*   **`llm_geolocation.py`**: Provides classes for performing geolocation tasks using Large Language Models (LLMs), supporting both external API-based models and local Hugging Face models.

## Module: `llm_geolocation.py`

*   **Main Role**: This module offers a high-level interface for obtaining geolocation predictions (latitude, longitude) from images, potentially augmented with contextual information (e.g., for RAG-based approaches). It handles the complexities of API interaction, model loading, prompt engineering (internal to the classes), and parsing responses.

*   **Key Classes**:

    1.  **`OpenAIGeolocator` Class**:
        *   **Purpose**: Designed to interact with OpenAI-compatible APIs (like OpenAI's own GPT-4 with Vision or other compatible endpoints) for geolocation tasks.
        *   **Functionalities**:
            *   Initializes with API credentials (API key, model name, base URL).
            *   Handles image encoding (e.g., base64 for API calls).
            *   Provides methods for both zero-shot (`get_openai_response`) and RAG-based (`get_openai_response_rag`) geolocation predictions. The RAG method takes candidate and reverse GPS coordinates as additional context.
            *   Includes a `run_predictions(args)` method that orchestrates the entire workflow: loading dataframes (with image IDs and potentially RAG candidate info), processing rows to get predictions, and saving the results. It supports different processing modes like 'predict', 'extract' (for parsing coordinates from raw LLM responses), 'rag', and 'rag_extract'.
            *   Utilizes `pandarallel` for batch processing of images if available.

    2.  **`HuggingFaceLlavaGeolocator` Class**:
        *   **Purpose**: Designed to use locally hosted Hugging Face LLaVA (or similar multi-modal) models for geolocation.
        *   **Functionalities**:
            *   Initializes by loading a specified LLaVA model and its processor from a local path, with options for `torch_dtype` and `device_map`.
            *   Provides methods for zero-shot (`get_hf_response`) and RAG-based (`get_hf_response_rag`) geolocation predictions using the local model.
            *   Includes a `run_predictions(args)` method, similar to the one in `OpenAIGeolocator`, to manage the end-to-end process of data loading, prediction generation (using `tqdm.pandas().progress_apply` for progress tracking), and result saving for different processing modes.

*   **Intended Usage**:
    *   The classes within `llm_geolocation.py` are not meant to be run directly. They are intended to be instantiated and controlled by dedicated launcher scripts.
    *   **`OpenAIGeolocator`** is typically used by `scripts/execute_openai_geoloc.py`.
    *   **`HuggingFaceLlavaGeolocator`** is typically used by `scripts/execute_hf_llava_geoloc.py`.
    *   These launcher scripts handle command-line argument parsing (API keys, model paths, data paths, processing modes, etc.) and then instantiate the appropriate geolocator class, finally calling its `run_predictions(args)` method.

    **Conceptual Example (from a launcher script for OpenAI):**
    ```python
    # In scripts/execute_openai_geoloc.py
    from services.llm_geolocation import OpenAIGeolocator
    # ... (argparse setup for args, including args.api_key, args.model_name, etc.) ...

    geolocator = OpenAIGeolocator(
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url
    )
    geolocator.run_predictions(args) # args contains all other necessary configurations
    ```

## Adding New Service-like Modules

1.  **Define the Service**: Clearly define the scope and functionality of the new service. Determine if it involves external APIs, complex local processing, or provides a distinct, reusable capability.
2.  **Create a New Module**: Add a new Python file to the `services/` directory (e.g., `my_new_service_client.py` or `advanced_data_processor_service.py`).
3.  **Implement the Class(es)**:
    *   Design one or more classes to encapsulate the service's logic.
    *   The `__init__` method should handle setup (e.g., API client initialization, loading local models or resources).
    *   Provide public methods that expose the service's capabilities in a clear and simple way.
    *   Consider adding a main orchestration method (like `run_predictions` or `execute_service`) that takes an `args` object or a configuration dictionary to manage the overall workflow if the service involves multiple steps or modes.
4.  **Error Handling and Configuration**: Implement robust error handling and make the service configurable through parameters passed to its constructor or methods.
5.  **Launcher Script**: Create a corresponding launcher script in the `scripts/` directory (e.g., `scripts/execute_my_new_service.py`). This script will:
    *   Use `argparse` to handle command-line arguments specific to configuring and running your new service.
    *   Import and instantiate the service class(es) from your new module in `services/`.
    *   Call the appropriate methods on the service instance.
6.  **Documentation**:
    *   Update this `README.md` to include a description of your new module and its class(es) under the "Current Structure" section.
    *   Briefly explain its role, key functionalities, and how it's intended to be used (mentioning its launcher script).
    *   Thoroughly document your new service class(es) and their methods with docstrings.
7.  **Dependencies**: If your new service introduces new external dependencies, ensure they are added to the project's `requirements.txt`.

By structuring complex functionalities as services in this directory, the project benefits from modularity, easier maintenance, and clearer separation of concerns.
