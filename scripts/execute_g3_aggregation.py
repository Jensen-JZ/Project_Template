# scripts/execute_g3_aggregation.py
import argparse
import os
import sys

# Adjust path to project root to find the 'dataprocessing' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dataprocessing.g3_prediction_aggregation import aggregate_llm_predictions
except ImportError as e:
    print(f"Error importing aggregate_llm_predictions: {e}")
    print("Ensure that dataprocessing/g3_prediction_aggregation.py exists and the PYTHONPATH is set correctly.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Execute G3 LLM Prediction Aggregation")

    # Default paths based on the original script's hardcoded values
    default_data_root = './data/im2gps3k/' # Assuming this is the common root for these files
    
    parser.add_argument('--raw_df_path', type=str, 
                        default=os.path.join(default_data_root, 'im2gps3k_places365.csv'),
                        help="Path to the raw dataframe CSV.")
    parser.add_argument('--zs_pred_path', type=str, 
                        default=os.path.join(default_data_root, 'llm_predict_results_zs.csv'),
                        help="Path to the zero-shot predictions CSV.")
    parser.add_argument('--rag5_pred_path', type=str, 
                        default=os.path.join(default_data_root, '5_llm_predict_results_rag.csv'),
                        help="Path to the RAG-5 predictions CSV.")
    parser.add_argument('--rag10_pred_path', type=str, 
                        default=os.path.join(default_data_root, '10_llm_predict_results_rag.csv'),
                        help="Path to the RAG-10 predictions CSV.")
    parser.add_argument('--rag15_pred_path', type=str, 
                        default=os.path.join(default_data_root, '15_llm_predict_results_rag.csv'),
                        help="Path to the RAG-15 predictions CSV.")
    parser.add_argument('--output_path', type=str, 
                        default=os.path.join(default_data_root, 'im2gps3k_prediction.csv'),
                        help="Path to save the aggregated CSV.")
    parser.add_argument('--n_predictions_per_response', type=int, default=10,
                        help="Number of predictions expected in each 'response' list within the prediction files (should match 'n_choices' during generation).")

    args = parser.parse_args()

    print("Starting G3 LLM Prediction Aggregation script...")
    print("Arguments received:")
    print(f"  Raw DataFrame Path: {args.raw_df_path}")
    print(f"  Zero-Shot Predictions Path: {args.zs_pred_path}")
    print(f"  RAG-5 Predictions Path: {args.rag5_pred_path}")
    print(f"  RAG-10 Predictions Path: {args.rag10_pred_path}")
    print(f"  RAG-15 Predictions Path: {args.rag15_pred_path}")
    print(f"  Output Path: {args.output_path}")
    print(f"  Predictions per response: {args.n_predictions_per_response}")

    aggregate_llm_predictions(args)
    
    print("G3 LLM Prediction Aggregation script finished.")

if __name__ == '__main__':
    main()
