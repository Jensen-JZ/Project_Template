# scripts/execute_g3_geo_ops.py
import argparse
import os
import sys
import pandas as pd
import numpy as np # For loading .npy files
import faiss # For reading index
import torch # Added for torch.cuda.is_available()

# Adjust path to project root to find the 'features' and 'utils' modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from features.g3_geospatial_operations import build_index, search_index, evaluate, res as faiss_gpu_resources
    # 'res' is the faiss.StandardGpuResources() initialized in g3_geospatial_operations
except ImportError as e:
    print(f"Error importing from features.g3_geospatial_operations: {e}")
    print("Ensure that features/g3_geospatial_operations.py exists and the PYTHONPATH is set correctly.")
    sys.exit(1)
except Exception as e: # Catch any other exception during import of features
    print(f"An unexpected error occurred during import from features.g3_geospatial_operations: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Execute G3 Geospatial Operations (Index, Search, Evaluate)")
    
    parser.add_argument('--index', type=str, default='g3', help="Type of index to build or use (e.g., 'g3').")
    parser.add_argument('--dataset', type=str, default='im2gps3k', help="Dataset to use for searching or evaluation (e.g., 'im2gps3k', 'yfcc4k').")
    parser.add_argument('--database', type=str, default='mp16', help="Database used for indexing (e.g., 'mp16' if G3 index was built on MP16).")
    
    parser.add_argument('--model_checkpoint_path', type=str, default='./checkpoints/g3.pth', help="Path to the G3 model checkpoint.")
    parser.add_argument('--mp16_root_path', type=str, default='./data/', help="Root path for MP16 dataset files.")
    parser.add_argument('--mp16_csv_path', type=str, default='MP16_Pro_filtered.csv', help="CSV file name for MP16 database (used in eval). Original index used MP16_Pro_places365.csv.")
    
    parser.add_argument('--im2gps3k_root_path', type=str, default='./data/im2gps3k', help="Root path for im2gps3k dataset.")
    parser.add_argument('--im2gps3k_csv_name', type=str, default='im2gps3k_places365.csv', help="CSV file name for im2gps3k dataset.")

    parser.add_argument('--yfcc4k_root_path', type=str, default='./data/yfcc4k', help="Root path for yfcc4k dataset.")
    parser.add_argument('--yfcc4k_csv_name', type=str, default='yfcc4k_places365.csv', help="CSV file name for yfcc4k dataset.")

    parser.add_argument('--llm_prediction_csv_path', type=str, default='', help="Path to LLM prediction CSV for reranking. If empty, uses default pattern: ./data/{dataset}/{dataset}_prediction.csv")
    parser.add_argument('--query_image_folder', type=str, default='', help="Path to query image folder for reranking. If empty, uses default pattern: ./data/{dataset}/images")
    parser.add_argument('--output_results_csv_path', type=str, default='', help="Path to save evaluation results CSV. If empty, uses default pattern: ./data/{dataset}_{index}_results.csv")

    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for DataLoader operations.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for DataLoader. Default 0 for PyTorch DataLoader (main process).")
    parser.add_argument('--topn_candidates_for_reranking', type=int, default=5, help="Top N candidates to use in LLM reranking for GeoImageDataset.")
    parser.add_argument('--search_topk', type=int, default=20, help="Top K neighbors to retrieve during FAISS search.")

    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    if faiss_gpu_resources is None and args.device == "cuda":
        print("Warning: CUDA device is available, but FAISS GPU resources failed to initialize in features module. FAISS will run on CPU if it's CPU-only FAISS build.")

    dataset_csv_full_path = ""
    if args.dataset == 'im2gps3k':
        dataset_csv_full_path = os.path.join(args.im2gps3k_root_path, args.im2gps3k_csv_name)
    elif args.dataset == 'yfcc4k':
        dataset_csv_full_path = os.path.join(args.yfcc4k_root_path, args.yfcc4k_csv_name)
    
    if os.path.exists(dataset_csv_full_path):
        args.dataset_df = pd.read_csv(dataset_csv_full_path)
        print(f"Loaded dataset_df from {dataset_csv_full_path}")
    else:
        args.dataset_df = None
        print(f"Warning: Dataset CSV {dataset_csv_full_path} not found for dataset '{args.dataset}'.")

    database_csv_full_path = ""
    if args.database == 'mp16':
        database_csv_full_path = os.path.join(args.mp16_root_path, args.mp16_csv_path)
    
    if os.path.exists(database_csv_full_path):
        args.database_df = pd.read_csv(database_csv_full_path)
        print(f"Loaded database_df from {database_csv_full_path}")
    else:
        args.database_df = None
        print(f"Warning: Database CSV {database_csv_full_path} not found for database '{args.database}'.")
    
    if not args.llm_prediction_csv_path:
        args.llm_prediction_csv_path = f'./data/{args.dataset}/{args.dataset}_prediction.csv'
    if not args.query_image_folder:
        args.query_image_folder = f'./data/{args.dataset}/images'
    if not args.output_results_csv_path:
        args.output_results_csv_path = f'./data/{args.dataset}_{args.index}_results.csv'

    index_dir = './index'
    index_file_path = os.path.join(index_dir, f'{args.index}.index')
    search_results_D_path = os.path.join(index_dir, f'D_{args.index}_{args.dataset}.npy')
    search_results_I_path = os.path.join(index_dir, f'I_{args.index}_{args.dataset}.npy')

    os.makedirs(index_dir, exist_ok=True)

    if not os.path.exists(index_file_path):
        print(f"Index file {index_file_path} not found. Building index...")
        if args.index == 'g3' and args.database != 'mp16':
             print(f"Warning: Building 'g3' index typically uses 'mp16' database. Current database is '{args.database}'. Ensure correct MP16 CSV for build is pointed to by --mp16_csv_path if it's different for build vs eval.")
        build_index(args)
    
    if os.path.exists(index_file_path):
        if not os.path.exists(search_results_I_path):
            print(f"Search results {search_results_I_path} not found. Performing search...")
            faiss_index_cpu = faiss.read_index(index_file_path)
            print(f"FAISS index {index_file_path} loaded on CPU successfully (ntotal={faiss_index_cpu.ntotal}).")
            
            faiss_index_to_use = faiss_index_cpu
            if args.device == "cuda" and faiss_gpu_resources is not None:
                print("Attempting to move FAISS index to GPU...")
                try:
                    gpu_id = 0 
                    faiss_index_gpu = faiss.index_cpu_to_gpu(faiss_gpu_resources, gpu_id, faiss_index_cpu)
                    faiss_index_to_use = faiss_index_gpu
                    print(f"FAISS index moved to GPU successfully (ntotal={faiss_index_to_use.ntotal}).")
                except Exception as e:
                    print(f"Error moving FAISS index to GPU: {e}. Using CPU index.")
            
            D, I = search_index(args, faiss_index_to_use, args.search_topk)
            if D is not None and I is not None:
                np.save(search_results_D_path, D)
                np.save(search_results_I_path, I)
                print(f"Search results saved to {search_results_D_path} and {search_results_I_path}")
                if args.dataset_df is not None and args.database_df is not None:
                    evaluate(args, I)
                else:
                    print("Skipping evaluation as dataset_df or database_df could not be loaded.")
            else:
                print("Search failed. Skipping saving results and evaluation.")
        else:
            print(f"Loading existing search results I from {search_results_I_path}...")
            I_loaded = np.load(search_results_I_path)
            # D_loaded = np.load(search_results_D_path) # D is not used by current evaluate
            if args.dataset_df is not None and args.database_df is not None:
                evaluate(args, I_loaded)
            else:
                print("Skipping evaluation as dataset_df or database_df could not be loaded.")
    else:
        print(f"Index file {index_file_path} still not found after attempting build. Exiting.")

if __name__ == '__main__':
    main()
