# scripts/execute_hf_llava_geoloc.py
import argparse
import os
import sys

# Adjust path to project root to find the 'services' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from services.llm_geolocation import HuggingFaceLlavaGeolocator, TQDM_AVAILABLE
    if TQDM_AVAILABLE:
        # from tqdm import tqdm # tqdm.pandas is initialized in the class method
        pass 
except ImportError as e:
    print(f"Error importing HuggingFaceLlavaGeolocator: {e}")
    print("Ensure that services/llm_geolocation.py exists and the PYTHONPATH is set correctly.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Execute HuggingFace LLaVA Geolocation Predictions")

    # Arguments from original g3_llm_predict_hf.py
    parser.add_argument('--model_path', type=str, required=True, help="Path to the HuggingFace LLaVA model directory.")
    
    # Paths
    parser.add_argument('--root_path', type=str, default="./data/im2gps3k", help="Root directory for data.")
    parser.add_argument('--text_path', type=str, default="im2gps3k_places365.csv", help="CSV file with image IDs.")
    parser.add_argument('--image_path', type=str, default="images", help="Subdirectory for images.")
    parser.add_argument('--result_path', type=str, default="llm_predict_results_zs_llava.csv", help="Output CSV for zero-shot predictions.")
    parser.add_argument('--rag_path', type=str, default="llm_predict_results_rag_llava.csv", help="Output CSV for RAG predictions.")
    parser.add_argument('--mp16_database_path', type=str, default='./data/MP16_Pro_filtered.csv', help="Path to MP16 database for RAG.")
    parser.add_argument('--searching_file_name', type=str, default='I_g3_im2gps3k', help="Base name for .npy RAG indices in './index/'.")

    # Process control
    parser.add_argument('--process', type=str, default='predict', choices=['predict', 'extract', 'rag', 'rag_extract'], help="Process type.")
    parser.add_argument('--rag_sample_num', type=int, default=5, help="Number of RAG samples.")
    parser.add_argument('--force_rag_preprocess', action='store_true', help="Force RAG preprocessing.")

    # HF specific model parameters (optional, will use defaults in HuggingFaceLlavaGeolocator if not provided)
    parser.add_argument('--torch_dtype_str', type=str, default='float16', help="Torch dtype for model (e.g., 'float16', 'bfloat16', 'float32').")
    parser.add_argument('--device_map', type=str, default='auto', help="Device map for model loading (e.g., 'auto', 'cuda:0').")
    
    # HF specific generation parameters (optional)
    parser.add_argument('--max_tokens', type=int, default=None, help="Max new tokens for HF model generation. Uses class default if None.")
    parser.add_argument('--temperature', type=float, default=None, help="Temperature for HF model sampling. Uses class default if None.")
    parser.add_argument('--n_choices', type=int, default=1, help="Number of choices for HF model (typically 1 as num_return_sequences > 1 needs specific model support). Uses class default if None.")

    # Performance / Debugging
    parser.add_argument('--tqdm_threshold', type=int, default=100, help="Dataframe row count above which tqdm.progress_apply is used.")
    parser.add_argument('--nrows_to_process', type=int, default=None, help="Number of rows to process from the input CSV. Processes all if None.")


    args = parser.parse_args()

    if TQDM_AVAILABLE:
        print("tqdm available. It will be initialized by HuggingFaceLlavaGeolocator if applicable.")
    else:
        print("tqdm not available. Progress bars for pandas apply will not be shown.")

    print(f"Instantiating HuggingFaceLlavaGeolocator with model: {args.model_path}")
    try:
        geolocator = HuggingFaceLlavaGeolocator(
            model_path=args.model_path,
            torch_dtype_str=args.torch_dtype_str,
            device_map=args.device_map
        )
    except Exception as e:
        print(f"Failed to initialize HuggingFaceLlavaGeolocator: {e}")
        sys.exit(1)
    
    print(f"Starting HF LLaVA geolocation process: {args.process}")
    geolocator.run_predictions(args)
    print(f"HF LLaVA geolocation process '{args.process}' finished.")

if __name__ == '__main__':
    main()
