# Services Directory (`services/`)

[English](#services-directory-services) | [中文](#services-目录-services)

## Services Directory (`services/`)

### Purpose

The `services/` directory is designed to house Python modules that encapsulate complex, self-contained functionalities, especially those that interact with external APIs or manage significant, distinct operational logic that acts like a "service." This includes, but is not limited to:

*   **External API Clients**: Modules that provide an interface to third-party services (e.g., cloud-based AI models, data providers).
*   **Complex Processing Units**: Components that manage intricate workflows involving multiple steps, model interactions, and data transformations, such as sophisticated prediction pipelines.
*   **Self-Contained Functionality**: Modules that can be used across different parts of the project or even potentially be decoupled for standalone use, offering a clear API to their capabilities.

The primary goal is to abstract away the internal complexity of these services, providing a clean and easy-to-use interface for other parts of the project, particularly launcher scripts in the `scripts/` directory.

### Current Structure

This directory currently contains the following key module:

*   **`llm_geolocation.py`**: Provides classes for performing geolocation tasks using Large Language Models (LLMs), supporting both external API-based models and local Hugging Face models.

### Module: `llm_geolocation.py`

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

### Adding New Service-like Modules

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

## Services 目录 (`services/`)

### 目的

`services/` 目录旨在容纳封装复杂、自包含功能的 Python 模块，特别是那些与外部 API 交互或管理重要、独立的运行逻辑（如同"服务"一样）的模块。这包括但不限于：

*   **外部 API 客户端**：提供第三方服务接口的模块（例如，基于云的 AI 模型、数据提供商）。
*   **复杂处理单元**：管理涉及多个步骤、模型交互和数据转换的复杂工作流的组件，比如复杂的预测流水线。
*   **自包含功能**：可以在项目的不同部分使用，甚至可能被解耦用于独立使用的模块，为其功能提供清晰的 API。

主要目标是抽象这些服务的内部复杂性，为项目的其他部分（特别是 `scripts/` 目录中的启动脚本）提供干净、易用的接口。

### 当前结构

该目录目前包含以下关键模块：

*   **`llm_geolocation.py`**：提供使用大型语言模型（LLMs）执行地理定位任务的类，支持基于外部 API 的模型和本地 Hugging Face 模型。

### 模块：`llm_geolocation.py`

*   **主要作用**：该模块提供了一个高级接口，用于从图像获取地理位置预测（纬度、经度），可能辅以上下文信息（例如，基于 RAG 的方法）。它处理 API 交互、模型加载、提示工程（类内部）和响应解析的复杂性。

*   **关键类**：

    1.  **`OpenAIGeolocator` 类**：
        *   **目的**：设计用于与 OpenAI 兼容的 API（如 OpenAI 自己的 GPT-4 with Vision 或其他兼容端点）进行地理定位任务交互。
        *   **功能**：
            *   使用 API 凭证（API 密钥、模型名称、基本 URL）初始化。
            *   处理图像编码（例如，用于 API 调用的 base64 编码）。
            *   提供零样本（`get_openai_response`）和基于 RAG 的（`get_openai_response_rag`）地理定位预测方法。RAG 方法将候选和反向 GPS 坐标作为额外上下文。
            *   包含一个 `run_predictions(args)` 方法，用于协调整个工作流：加载数据帧（带有图像 ID 和可能的 RAG 候选信息）、处理行以获取预测并保存结果。它支持不同的处理模式，如 'predict'、'extract'（用于从原始 LLM 响应中解析坐标）、'rag' 和 'rag_extract'。
            *   如果可用，利用 `pandarallel` 进行图像的批处理。

    2.  **`HuggingFaceLlavaGeolocator` 类**：
        *   **目的**：设计用于使用本地托管的 Hugging Face LLaVA（或类似多模态）模型进行地理定位。
        *   **功能**：
            *   通过从本地路径加载指定的 LLaVA 模型及其处理器初始化，带有 `torch_dtype` 和 `device_map` 选项。
            *   使用本地模型提供零样本（`get_hf_response`）和基于 RAG 的（`get_hf_response_rag`）地理定位预测方法。
            *   包含一个 `run_predictions(args)` 方法，类似于 `OpenAIGeolocator` 中的方法，用于管理数据加载、预测生成（使用 `tqdm.pandas().progress_apply` 进行进度跟踪）和结果保存的端到端过程，适用于不同的处理模式。

*   **预期用法**：
    *   `llm_geolocation.py` 中的类不应直接运行。它们旨在由专用的启动脚本实例化和控制。
    *   **`OpenAIGeolocator`** 通常由 `scripts/execute_openai_geoloc.py` 使用。
    *   **`HuggingFaceLlavaGeolocator`** 通常由 `scripts/execute_hf_llava_geoloc.py` 使用。
    *   这些启动脚本处理命令行参数解析（API 密钥、模型路径、数据路径、处理模式等），然后实例化适当的地理定位器类，最后调用其 `run_predictions(args)` 方法。

    **概念示例（来自 OpenAI 的启动脚本）：**
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

### 添加新的服务类模块

1.  **定义服务**：明确定义新服务的范围和功能。确定它是否涉及外部 API、复杂的本地处理，或提供独特的可重用功能。
2.  **创建新模块**：向 `services/` 目录添加新的 Python 文件（例如，`my_new_service_client.py` 或 `advanced_data_processor_service.py`）。
3.  **实现类**：
    *   设计一个或多个类来封装服务的逻辑。
    *   `__init__` 方法应处理设置（例如，API 客户端初始化，加载本地模型或资源）。
    *   提供公共方法，以清晰简单的方式展示服务的功能。
    *   如果服务涉及多个步骤或模式，考虑添加一个主协调方法（如 `run_predictions` 或 `execute_service`），接受 `args` 对象或配置字典来管理整体工作流。
4.  **错误处理和配置**：实现健壮的错误处理，并通过传递给其构造函数或方法的参数使服务可配置。
5.  **启动脚本**：在 `scripts/` 目录中创建相应的启动脚本（例如，`scripts/execute_my_new_service.py`）。该脚本将：
    *   使用 `argparse` 处理特定于配置和运行新服务的命令行参数。
    *   从 `services/` 中的新模块导入并实例化服务类。
    *   调用服务实例上的适当方法。
6.  **文档**：
    *   更新此 `README.md`，在"当前结构"部分包含对新模块及其类的描述。
    *   简要解释其角色、关键功能以及它的预期用法（提及其启动脚本）。
    *   使用文档字符串彻底记录您的新服务类及其方法。
7.  **依赖关系**：如果您的新服务引入了新的外部依赖，请确保将它们添加到项目的 `requirements.txt` 中。

通过将复杂功能构建为此目录中的服务，项目受益于模块化、更易维护和更清晰的关注点分离。
