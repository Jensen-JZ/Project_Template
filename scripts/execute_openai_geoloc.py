# scripts/execute_openai_geoloc.py
import argparse
import os
import sys

# Adjust path to project root to find the 'services' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from services.llm_geolocation import OpenAIGeolocator, PANDARALLEL_AVAILABLE
    if PANDARALLEL_AVAILABLE:
        # from pandarallel import pandarallel # Initialization is handled within the class
        pass
except ImportError as e:
    print(f"Error importing OpenAIGeolocator: {e}")
    print("Ensure that services/llm_geolocation.py exists and the PYTHONPATH is set correctly.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Execute OpenAI Geolocation Predictions")
    
    # API related
    parser.add_argument('--api_key', type=str, required=True, help="OpenAI API key.")
    parser.add_argument('--model_name', type=str, default="gpt-4-vision-preview", help="OpenAI model name (e.g., gpt-4-vision-preview).")
    parser.add_argument('--base_url', type=str, default="https://api.openai.com/v1/chat/completions", help="OpenAI API base URL.")
    
    # Paths
    parser.add_argument('--root_path', type=str, default="./data/im2gps3k", help="Root directory for data (text file, images).")
    parser.add_argument('--text_path', type=str, default="im2gps3k_places365.csv", help="Name of the CSV file containing image IDs and other metadata, relative to root_path.")
    parser.add_argument('--image_path', type=str, default="images", help="Subdirectory name for images, relative to root_path.")
    parser.add_argument('--result_path', type=str, default="llm_predict_results_zs.csv", help="Name of the output CSV file for zero-shot predictions, relative to root_path.")
    parser.add_argument('--rag_path', type=str, default="llm_predict_results_rag.csv", help="Name of the output CSV file for RAG predictions, relative to root_path.")
    parser.add_argument('--mp16_database_path', type=str, default='./data/MP16_Pro_filtered.csv', help="Path to the MP16 database CSV for RAG candidates.")
    parser.add_argument('--searching_file_name', type=str, default='I_g3_im2gps3k', help="Base name of the .npy files (without _reverse.npy or .npy) for RAG candidate indices, expected in './index/'.")

    # Process control
    parser.add_argument('--process', type=str, default='predict', choices=['predict', 'extract', 'rag', 'rag_extract'], help="Type of process to run.")
    parser.add_argument('--rag_sample_num', type=int, default=15, help="Number of candidate samples for RAG.")
    parser.add_argument('--force_rag_preprocess', action='store_true', help="Force RAG preprocessing even if RAG file exists.")
    
    # OpenAI specific request parameters (optional, will use defaults in OpenAIGeolocator if not provided)
    parser.add_argument('--detail', type=str, default=None, help="Detail level for OpenAI image processing (e.g., 'low', 'high', 'auto'). Uses class default if None.")
    parser.add_argument('--max_tokens', type=int, default=None, help="Max tokens for OpenAI response. Uses class default if None.")
    parser.add_argument('--temperature', type=float, default=None, help="Temperature for OpenAI sampling. Uses class default if None.")
    parser.add_argument('--n_choices', type=int, default=None, help="Number of choices (response candidates) from OpenAI. Uses class default if None.")

    # Performance / Debugging
    parser.add_argument('--num_pandarallel_workers', type=int, default=4, help="Number of workers for pandarallel.")
    parser.add_argument('--pandarallel_threshold', type=int, default=100, help="Dataframe row count above which pandarallel is used.")
    parser.add_argument('--nrows_to_process', type=int, default=None, help="Number of rows to process from the input CSV (for debugging/testing). Processes all if None.")

    args = parser.parse_args()

    if PANDARALLEL_AVAILABLE:
        print(f"Pandarallel available. It will be initialized by OpenAIGeolocator with {args.num_pandarallel_workers} workers if applicable.")
    else:
        print("Pandarallel not available. Processing will be sequential.")

    print(f"Instantiating OpenAIGeolocator for model: {args.model_name}")
    geolocator = OpenAIGeolocator(
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url
    )
    
    print(f"Starting geolocation process: {args.process}")
    geolocator.run_predictions(args)
    print(f"Geolocation process '{args.process}' finished.")

if __name__ == '__main__':
    main()
