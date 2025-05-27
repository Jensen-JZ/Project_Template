# dataprocessing/g3_prediction_aggregation.py
import pandas as pd
import re
import ast
from tqdm import tqdm
import os # For checking file existence

def aggregate_llm_predictions(args):
    """
    Aggregates LLM predictions from various CSV files into a single raw dataframe.

    Args:
        args: An object (e.g., from argparse) containing file paths:
              - args.raw_df_path: Path to the raw dataframe CSV.
              - args.zs_pred_path: Path to the zero-shot predictions CSV.
              - args.rag5_pred_path: Path to the RAG-5 predictions CSV.
              - args.rag10_pred_path: Path to the RAG-10 predictions CSV.
              - args.rag15_pred_path: Path to the RAG-15 predictions CSV.
              - args.output_path: Path to save the aggregated CSV.
              - args.n_predictions_per_response (optional, default 10): Number of predictions
                expected in each 'response' list within the prediction files.
    """
    print("Starting LLM prediction aggregation...")

    required_files = {
        "raw_df_path": args.raw_df_path,
        "zs_pred_path": args.zs_pred_path,
        "rag5_pred_path": args.rag5_pred_path,
        "rag10_pred_path": args.rag10_pred_path,
        "rag15_pred_path": args.rag15_pred_path
    }
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"Error: Input file not found for {name} at path: '{path}'")
            return
    
    print(f"Loading raw dataframe from: {args.raw_df_path}")
    df_raw = pd.read_csv(args.raw_df_path)
    
    print(f"Loading zero-shot predictions from: {args.zs_pred_path}")
    zs_df = pd.read_csv(args.zs_pred_path)
    
    print(f"Loading RAG-5 predictions from: {args.rag5_pred_path}")
    rag_5_df = pd.read_csv(args.rag5_pred_path)
    
    print(f"Loading RAG-10 predictions from: {args.rag10_pred_path}")
    rag_10_df = pd.read_csv(args.rag10_pred_path)
    
    print(f"Loading RAG-15 predictions from: {args.rag15_pred_path}")
    rag_15_df = pd.read_csv(args.rag15_pred_path)

    pattern = r'[-+]?\d+\.\d+' 
    n_preds_per_response = getattr(args, 'n_predictions_per_response', 10) 

    print(f"Processing zero-shot predictions (expecting up to {n_preds_per_response} per entry)...")
    for i in tqdm(range(zs_df.shape[0]), desc="Aggregating ZS Preds"):
        if i >= len(df_raw):
            break
        try:
            response_list_str = zs_df.loc[i, 'response']
            response_actual_list = ast.literal_eval(response_list_str) 
            
            for idx, content_json_str in enumerate(response_actual_list):
                if idx >= n_preds_per_response: break 
                try:
                    coords = ast.literal_eval(content_json_str) 
                    latitude = coords.get('latitude', 0.0)
                    longitude = coords.get('longitude', 0.0)
                except (ValueError, SyntaxError, TypeError, AttributeError): 
                    match = re.findall(pattern, str(content_json_str)) # Ensure content_json_str is string for regex
                    latitude = match[0] if len(match) >= 1 else '0.0'
                    longitude = match[1] if len(match) >= 2 else '0.0'
                
                df_raw.loc[i, f'zs_{idx}_latitude'] = latitude
                df_raw.loc[i, f'zs_{idx}_longitude'] = longitude
        except Exception as e:
            # print(f"Error processing row {i} in zs_df: {e}. Assigning defaults.") # Reduce noise
            for idx_default in range(n_preds_per_response):
                df_raw.loc[i, f'zs_{idx_default}_latitude'] = '0.0'
                df_raw.loc[i, f'zs_{idx_default}_longitude'] = '0.0'

    dataframes_and_prefixes = [
        (rag_5_df, "5_rag"),
        (rag_10_df, "10_rag"),
        (rag_15_df, "15_rag")
    ]

    for rag_df, prefix in dataframes_and_prefixes:
        print(f"Processing {prefix} predictions (expecting up to {n_preds_per_response} per entry)...")
        for i in tqdm(range(rag_df.shape[0]), desc=f"Aggregating {prefix} Preds"):
            if i >= len(df_raw): 
                break
            try:
                response_list_str = rag_df.loc[i, 'rag_response'] 
                response_actual_list = ast.literal_eval(response_list_str)

                for idx, content_json_str in enumerate(response_actual_list):
                    if idx >= n_preds_per_response: break
                    try:
                        coords = ast.literal_eval(content_json_str)
                        latitude = coords.get('latitude', 0.0)
                        longitude = coords.get('longitude', 0.0)
                    except (ValueError, SyntaxError, TypeError, AttributeError):
                        match = re.findall(pattern, str(content_json_str))
                        latitude = match[0] if len(match) >=1 else '0.0'
                        longitude = match[1] if len(match) >=2 else '0.0'
                    
                    df_raw.loc[i, f'{prefix}_{idx}_latitude'] = latitude
                    df_raw.loc[i, f'{prefix}_{idx}_longitude'] = longitude
            except Exception as e:
                # print(f"Error processing row {i} in {prefix}_df: {e}. Assigning defaults.") # Reduce noise
                for idx_default in range(n_preds_per_response):
                    df_raw.loc[i, f'{prefix}_{idx_default}_latitude'] = '0.0'
                    df_raw.loc[i, f'{prefix}_{idx_default}_longitude'] = '0.0'
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    df_raw.to_csv(args.output_path, index=False)
    print(f"Aggregated predictions saved to: {args.output_path}")
    print("Aggregation complete.")
