# features/g3_geospatial_operations.py
import faiss
import torch
import numpy as np
import os
import pandas as pd
from PIL import Image # Used in GeoImageDataset
from geopy.distance import geodesic # Used in evaluate (indirectly, if evaluate logic were more complex, but currently not directly used)
from torch.utils.data import Dataset, DataLoader # Dataset for GeoImageDataset, DataLoader for build/search
from tqdm import tqdm

# Attempt to import dataset classes from utils.g3_utils
try:
    from utils.g3_utils import MP16Dataset, im2gps3kDataset, yfcc4kDataset
except ImportError:
    import sys
    # Adjust path to project root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.g3_utils import MP16Dataset, im2gps3kDataset, yfcc4kDataset

# Initialize FAISS GPU resources globally for this module
# Note: This makes the module stateful regarding GPU resources.
# If multiple parts of a larger application used this, careful resource management might be needed.
# For this specific refactoring, it mirrors the original script's single resource object.
try:
    res = faiss.StandardGpuResources()
    print("FAISS StandardGpuResources initialized.")
except AttributeError:
    print("FAISS GPU resources (StandardGpuResources) not available. Operations requiring GPU FAISS will fail.")
    res = None # Allow CPU FAISS to still work if GPU is not compiled in.
except Exception as e:
    print(f"Error initializing FAISS StandardGpuResources: {e}")
    res = None


class GeoImageDataset(Dataset):
    def __init__(self, dataframe, img_folder, topn, vision_processor, database_df, I):
        self.dataframe = dataframe.reset_index(drop=True) # Ensure dataframe has a simple 0-based index
        self.img_folder = img_folder
        self.topn = topn
        self.vision_processor = vision_processor
        self.database_df = database_df
        self.I = I # Search results (indices)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Ensure idx is an integer, not a tensor, if coming from DataLoader with certain samplers
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        img_path = f'{self.img_folder}/{self.dataframe.loc[idx, "IMG_ID"]}'
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(3,224,224)
        except Exception as e:
            print(f"Error loading image {img_path} for index {idx}: {e}. Returning dummy image.")
            image = torch.zeros(3, 224, 224) # Dummy image

        gps_data = []
        try:
            search_top1_latitude, search_top1_longitude = self.database_df.loc[self.I[idx][0], ['LAT', 'LON']].values
        except Exception as e:
            print(f"Error accessing database_df or I for query index {idx} (retrieved DB index {self.I[idx][0]}): {e}. Using dummy GPS for top1.")
            search_top1_latitude, search_top1_longitude = 0.0, 0.0
            
        for j in range(self.topn):
            try:
                gps_data.extend([
                    float(self.dataframe.loc[idx, f'5_rag_{j}_latitude']),
                    float(self.dataframe.loc[idx, f'5_rag_{j}_longitude']),
                    float(self.dataframe.loc[idx, f'10_rag_{j}_latitude']),
                    float(self.dataframe.loc[idx, f'10_rag_{j}_longitude']),
                    float(self.dataframe.loc[idx, f'15_rag_{j}_latitude']),
                    float(self.dataframe.loc[idx, f'15_rag_{j}_longitude']),
                    float(self.dataframe.loc[idx, f'zs_{j}_latitude']),
                    float(self.dataframe.loc[idx, f'zs_{j}_longitude']),
                    search_top1_latitude,
                    search_top1_longitude
                ])
            except KeyError as e: 
                # print(f"Warning: Missing RAG data for index {idx}, column {e}. Using dummy values (0,0).")
                gps_data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, search_top1_latitude, search_top1_longitude])
            except Exception as e:
                # print(f"Unexpected error processing GPS data for index {idx} at RAG level {j}: {e}. Using dummy values.")
                gps_data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, search_top1_latitude, search_top1_longitude])

        gps_data = np.array(gps_data).reshape(-1, 2)
        return image, gps_data, idx # Return original dataframe index for mapping results back


def build_index(args):
    model_checkpoint_path = getattr(args, 'model_checkpoint_path', './checkpoints/g3.pth')
    index_save_path = f'./index/{args.index}.index'

    print(f"Building index: {args.index}")
    print(f"Using model checkpoint: {model_checkpoint_path}")
    print(f"Target index file: {index_save_path}")

    if not os.path.exists(model_checkpoint_path):
        print(f"ERROR: Model checkpoint {model_checkpoint_path} not found.")
        return

    model = torch.load(model_checkpoint_path, map_location=args.device)
    model.requires_grad_(False)
    model.eval()

    if args.index == 'g3':
        print("Using MP16Dataset for index building (as implied for 'g3' index).")
        dataset = MP16Dataset(vision_processor=model.vision_processor, text_processor=None, root_path=getattr(args, 'mp16_root_path', './data/'), text_data_path=getattr(args, 'mp16_csv_path', 'MP16_Pro_places365.csv'))
    else:
        print(f"Warning: Index type '{args.index}' does not have a specified dataset for building. This may fail if not MP16.")
        dataset = MP16Dataset(vision_processor=model.vision_processor, text_processor=None, root_path=getattr(args, 'mp16_root_path', './data/'), text_data_path=getattr(args, 'mp16_csv_path', 'MP16_Pro_places365.csv'))

    index_flat = faiss.IndexFlatIP(768*3)
    dataloader = DataLoader(dataset, batch_size=getattr(args, 'batch_size', 1024), shuffle=False, num_workers=getattr(args, 'num_workers', 0), pin_memory=True, prefetch_factor=3 if getattr(args, 'num_workers', 0) > 0 else None)
    
    t = tqdm(dataloader, desc="Building Index")
    for i, (images, texts, longitude, latitude) in enumerate(t):
        images = images.to(args.device)
        with torch.no_grad():
            vision_output = model.vision_model(images)[1]
            image_embeds = model.vision_projection(vision_output)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
            image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)
            image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
            image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)
            concatenated_embeds = torch.cat([image_embeds, image_text_embeds, image_location_embeds], dim=1)
        index_flat.add(concatenated_embeds.cpu().detach().numpy())

    print(f"Index built with {index_flat.ntotal} vectors.")
    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
    faiss.write_index(index_flat, index_save_path)
    print(f"Index saved to {index_save_path}")


def search_index(args, index, topk):
    model_checkpoint_path = getattr(args, 'model_checkpoint_path', './checkpoints/g3.pth')
    print(f"Searching index: {args.index} for dataset: {args.dataset} with topk: {topk}")
    print(f"Using model checkpoint: {model_checkpoint_path}")
    
    if not os.path.exists(model_checkpoint_path):
        print(f"ERROR: Model checkpoint {model_checkpoint_path} not found.")
        return None, None

    model = torch.load(model_checkpoint_path, map_location=args.device)
    model.requires_grad_(False)
    model.eval()
    vision_processor = model.vision_processor
    
    if args.dataset == 'im2gps3k':
        print("Using im2gps3kDataset for searching.")
        dataset = im2gps3kDataset(vision_processor=vision_processor, text_processor=None, root_path=getattr(args, 'im2gps3k_root_path', './data/im2gps3k'))
    elif args.dataset == 'yfcc4k':
        print("Using yfcc4kDataset for searching.")
        dataset = yfcc4kDataset(vision_processor=vision_processor, text_processor=None, root_path=getattr(args, 'yfcc4k_root_path', './data/yfcc4k'))
    else:
        print(f"ERROR: Dataset '{args.dataset}' is not supported for searching.")
        return None, None
        
    dataloader = DataLoader(dataset, batch_size=getattr(args, 'batch_size', 256), shuffle=False, num_workers=getattr(args, 'num_workers', 0), pin_memory=True, prefetch_factor=5 if getattr(args, 'num_workers', 0) > 0 else None)
    
    test_images_embeds_list = []
    print('Generating embeddings for search queries...')
    t = tqdm(dataloader, desc=f"Generating Embeddings for {args.dataset}")
    for i, (images, texts, longitude, latitude) in enumerate(t):
        images = images.to(args.device)
        with torch.no_grad():
            vision_output = model.vision_model(images)[1]
            if args.index == 'g3':
                image_embeds_part1 = model.vision_projection(vision_output)
                image_embeds_part1 = image_embeds_part1 / image_embeds_part1.norm(p=2, dim=-1, keepdim=True)
                image_text_embeds = model.vision_projection_else_1(model.vision_projection(vision_output))
                image_text_embeds = image_text_embeds / image_text_embeds.norm(p=2, dim=-1, keepdim=True)
                image_location_embeds = model.vision_projection_else_2(model.vision_projection(vision_output))
                image_location_embeds = image_location_embeds / image_location_embeds.norm(p=2, dim=-1, keepdim=True)
                current_embeds = torch.cat([image_embeds_part1, image_text_embeds, image_location_embeds], dim=1)
            else:
                print(f"Warning: Index type '{args.index}' search behavior not fully specified. Using default G3 vision projection.")
                current_embeds = model.vision_projection(vision_output)
                current_embeds = current_embeds / current_embeds.norm(p=2, dim=-1, keepdim=True)
        test_images_embeds_list.append(current_embeds.cpu().detach().numpy())
    
    test_images_embeds = np.concatenate(test_images_embeds_list, axis=0)
    print(f"Total query embeddings generated: {test_images_embeds.shape}")
    print('Starting FAISS search...')
    D, I = index.search(test_images_embeds, topk)
    print(f"Search complete. Found {I.shape[0]} results with {I.shape[1]} neighbors each.")
    return D, I


def evaluate(args, I):
    model_checkpoint_path = getattr(args, 'model_checkpoint_path', './checkpoints/g3.pth')
    print(f"Evaluating for dataset: {args.dataset} using index type: {args.index}")
    print(f"Using model checkpoint: {model_checkpoint_path}")

    if args.database_df is None or args.dataset_df is None:
        print("ERROR: Database or dataset DataFrame not provided in args for evaluation.")
        return

    if not os.path.exists(model_checkpoint_path):
        print(f"ERROR: Model checkpoint {model_checkpoint_path} not found.")
        return

    database = args.database_df
    df = args.dataset_df.copy().reset_index(drop=True)

    df['NN_idx'] = I[:, 0]
    # Ensure NN_idx is valid for database.loc
    valid_nn_idx = df['NN_idx'] < len(database)
    df.loc[valid_nn_idx, 'LAT_pred'] = database.loc[df.loc[valid_nn_idx, 'NN_idx'], 'LAT'].values
    df.loc[valid_nn_idx, 'LON_pred'] = database.loc[df.loc[valid_nn_idx, 'NN_idx'], 'LON'].values
    if not valid_nn_idx.all():
        print(f"Warning: {sum(~valid_nn_idx)} NN_idx values were out of bounds for the database_df. These predictions will be NaN.")
        df.loc[~valid_nn_idx, ['LAT_pred', 'LON_pred']] = np.nan


    llm_prediction_csv_path = getattr(args, 'llm_prediction_csv_path', f'./data/{args.dataset}/{args.dataset}_prediction.csv')
    print(f"Attempting to load LLM predictions for reranking from: {llm_prediction_csv_path}")

    if not os.path.exists(llm_prediction_csv_path):
        print(f"Warning: LLM prediction file {llm_prediction_csv_path} not found. Skipping LLM reranking.")
    else:
        df_llm = pd.read_csv(llm_prediction_csv_path).reset_index(drop=True)
        # Align df_llm with df (dataset_df) based on IMG_ID if they are not identical
        if not df['IMG_ID'].equals(df_llm['IMG_ID']):
             print("Warning: IMG_ID columns in dataset_df and df_llm are not identical or not in the same order. Attempting merge for alignment.")
             # Keep track of original df order
             df_original_order_idx = df.index.name if df.index.name else 'original_df_idx'
             if df_original_order_idx in df.columns: # avoid name collision
                 df_original_order_idx = f"{df_original_order_idx}_temp" 
             df[df_original_order_idx] = df.index
             # Merge df_llm into df based on IMG_ID
             df = pd.merge(df, df_llm, on="IMG_ID", how="left", suffixes=("", "_llm"))
             # Check if all original images found a match in df_llm
             if df[f'5_rag_0_latitude_llm'].isnull().any(): # Check a RAG col from df_llm
                 print("Warning: Some images in dataset_df did not find matching RAG data in df_llm after merge.")
             # Rename _llm columns to what GeoImageDataset expects
             for col_base in ['5_rag', '10_rag', '15_rag', 'zs']:
                 for j_val in range(getattr(args, 'topn_candidates_for_reranking', 5)): # Use a reasonable default for topn
                     for coord_suffix in ['latitude', 'longitude']:
                        if f'{col_base}_{j_val}_{coord_suffix}_llm' in df.columns:
                           df.rename(columns={f'{col_base}_{j_val}_{coord_suffix}_llm': f'{col_base}_{j_val}_{coord_suffix}'}, inplace=True)
             df_llm_for_dataset = df # Use the merged and potentially reordered df
        else:
            df_llm_for_dataset = df_llm


        model = torch.load(model_checkpoint_path, map_location=args.device)
        model.eval()
        
        topn_candidates_for_reranking = getattr(args, 'topn_candidates_for_reranking', 5)
        
        required_rag_cols_exist = all(f'5_rag_0_latitude' in df_llm_for_dataset.columns for _ in '_') # Simplified check

        if not required_rag_cols_exist:
            print(f"Warning: df_llm (from {llm_prediction_csv_path}) does not seem to contain the required RAG columns (e.g., '5_rag_0_latitude') after potential merge. Skipping LLM reranking.")
        else:
            print("Proceeding with LLM reranking.")
            query_image_folder = getattr(args, 'query_image_folder', f'./data/{args.dataset}/images')
            if not os.path.isdir(query_image_folder):
                print(f"ERROR: Query image folder {query_image_folder} not found for GeoImageDataset. Skipping LLM reranking.")
            else:
                # Pass df_llm_for_dataset to GeoImageDataset
                geo_image_dataset = GeoImageDataset(df_llm_for_dataset, query_image_folder, topn_candidates_for_reranking, vision_processor=model.vision_processor, database_df=database, I=I)
                geo_image_dataloader = DataLoader(geo_image_dataset, batch_size=getattr(args, 'batch_size', 256), shuffle=False, num_workers=getattr(args, 'num_workers', 0))

                for images, gps_batch, original_indices_from_geo_dataset in tqdm(geo_image_dataloader, desc="LLM Reranking"):
                    images = images.to(args.device)
                    with torch.no_grad():
                        image_embeds = model.vision_projection_else_2(model.vision_projection(model.vision_model(images)[1]))
                        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                        gps_batch = gps_batch.to(args.device).float()
                        b, c, _ = gps_batch.shape
                        gps_input = gps_batch.reshape(b*c, 2)
                        location_embeds = model.location_encoder(gps_input)
                        location_embeds = model.location_projection_else(location_embeds.reshape(b*c, -1))
                        location_embeds = location_embeds / location_embeds.norm(p=2, dim=-1, keepdim=True)
                        location_embeds = location_embeds.reshape(b, c, -1)
                        similarity = torch.matmul(image_embeds.unsqueeze(1), location_embeds.permute(0, 2, 1))
                        similarity = similarity.squeeze(1).cpu().detach().numpy()
                        max_idxs_in_candidates = np.argmax(similarity, axis=1)
                    
                    for i_batch, max_idx_in_candidate_set in enumerate(max_idxs_in_candidates):
                        # original_df_idx is the index in df_llm_for_dataset (which is aligned with df)
                        original_df_idx = original_indices_from_geo_dataset[i_batch].item()
                        final_latitude, final_longitude = gps_batch[i_batch][max_idx_in_candidate_set].cpu().numpy()
                        
                        if -90 <= final_latitude <= 90 and -180 <= final_longitude <= 180:
                            df.loc[original_df_idx, 'LAT_pred'] = final_latitude
                            df.loc[original_df_idx, 'LON_pred'] = final_longitude
                        else:
                            print(f"Warning: Invalid coordinates ({final_latitude}, {final_longitude}) from LLM reranking for original index {original_df_idx}. Retaining previous prediction.")
            # If merge happened, restore original df order
            if 'df_original_order_idx' in locals() and df_original_order_idx in df.columns:
                df.set_index(df_original_order_idx, inplace=True)
                df.sort_index(inplace=True)
                df.drop(columns=[df_original_order_idx], inplace=True, errors='ignore')


    df['geodesic'] = df.apply(lambda x: geodesic((x['LAT'], x['LON']), (x['LAT_pred'], x['LON_pred'])).km if pd.notnull(x['LAT']) and pd.notnull(x['LON']) and pd.notnull(x['LAT_pred']) and pd.notnull(x['LON_pred']) else np.nan, axis=1)
    
    output_results_csv_path = getattr(args, 'output_results_csv_path', f'./data/{args.dataset}_{args.index}_results.csv')
    df.to_csv(output_results_csv_path, index=False)
    print(f"Evaluation results saved to {output_results_csv_path}")

    if 'geodesic' in df.columns and not df['geodesic'].isnull().all():
        print('Accuracy at various distance thresholds:')
        valid_geodesic_count = df['geodesic'].notna().sum()
        if valid_geodesic_count > 0:
            for km_threshold in [1, 25, 200, 750, 2500]:
                accuracy = df[df['geodesic'] < km_threshold].shape[0] / valid_geodesic_count
                print(f'  < {km_threshold} km: {accuracy*100:.2f}% ({df[df["geodesic"] < km_threshold].shape[0]}/{valid_geodesic_count})')
        else:
            print("No valid geodesic distances to calculate accuracy.")
    else:
        print("Geodesic distances could not be calculated for any entries.")
