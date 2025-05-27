# services/llm_geolocation.py
import os
import sys
import base64
import re
import ast
import json # For HF part, not explicitly in OpenAI part but good for consistency
import time # For HF part

# Third-party imports
import pandas as pd
import numpy as np
import requests # For OpenAIGeolocator

try:
    from pandarallel import pandarallel
    PANDARALLEL_AVAILABLE = True
except ImportError:
    PANDARALLEL_AVAILABLE = False
    # print("Warning: pandarallel not found. Some operations will run sequentially.") # Reduce noise

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # print("Warning: tqdm not found. Progress bars will not be shown.") # Reduce noise

# PyTorch and Transformers imports (primarily for HuggingFaceLlavaGeolocator)
try:
    import torch
    from PIL import Image
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # print("Warning: PyTorch or Transformers not found. HuggingFaceLlavaGeolocator will not be available.") # Reduce noise

# Helper function (can be outside classes or static if preferred)
def _parse_llm_response_to_coords(response_content_str):
    """
    Parses a string assumed to be a JSON like '{"latitude": float, "longitude": float}'
    into a list of two floats. Returns [0.0, 0.0] on failure.
    """
    try:
        # Try direct JSON parsing first
        coords_dict = json.loads(response_content_str)
        lat = float(coords_dict.get("latitude", 0.0))
        lon = float(coords_dict.get("longitude", 0.0))
        return [lat, lon]
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback to regex if direct JSON parsing fails or if it's not a dict
        pattern = r'[-+]?\d+\.\d+' # Matches float numbers
        try:
            matches = re.findall(pattern, response_content_str)
            if len(matches) >= 2:
                return [float(matches[0]), float(matches[1])]
        except (TypeError, ValueError):
            pass # Error in regex finding or float conversion
    return [0.0, 0.0] # Default on any parsing failure


class OpenAIGeolocator:
    def __init__(self, api_key, model_name, base_url, default_detail="low", default_max_tokens=200, default_temperature=1.2, default_n_choices=10):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.default_detail = default_detail
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_n_choices = default_n_choices
        # print(f"OpenAIGeolocator initialized for model: {model_name}") # Reduce noise

    @staticmethod
    def _encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            # print(f"Error encoding image {image_path}: {e}") # Reduce noise
            return None

    def get_openai_response(self, image_path, detail=None, max_tokens=None, temperature=None, n=None):
        detail = detail or self.default_detail
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        n = n or self.default_n_choices
        
        base64_image = self._encode_image(image_path)
        if not base64_image: return ['{"latitude": 0.0,"longitude": 0.0}'] * n

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": """Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image.
                    Please give me the location of the given image.
                    Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.
                    Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}.
                    Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}."""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}}
                ]}
            ], "max_tokens": max_tokens, "temperature": temperature, "n": n
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=(30, 60))
            response.raise_for_status() 
            ans = [choice['message']['content'] for choice in response.json().get('choices', []) if choice.get('message')]
            if not ans: return ['{"latitude": 0.0,"longitude": 0.0}'] * n 
            while len(ans) < n: ans.append(ans[0] if ans else '{"latitude": 0.0,"longitude": 0.0}') 
            return ans[:n]
        except requests.RequestException as e:
            # print(f"API request failed: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * n
        except Exception as e:
            # print(f"Error processing OpenAI response: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * n


    def get_openai_response_rag(self, image_path, candidates_gps, reverse_gps, detail=None, max_tokens=None, temperature=None, n=None):
        detail = detail or self.default_detail
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        n = n or self.default_n_choices

        base64_image = self._encode_image(image_path)
        if not base64_image: return ['{"latitude": 0.0,"longitude": 0.0}'] * n
        
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""Suppose you are an expert in geo-localization, Please analyze this image and give me a guess of the location.
                    Your answer must be to the coordinates level in (latitude, longitude) format.
                    For your reference, these are coordinates of some similar images: {candidates_gps}, and these are coordinates of some dissimilar images: {reverse_gps}.
                    Remember, you must have an answer, just output your best guess, don't answer me that you can't give an location.
                    Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}.
                    Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}."""},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": detail}}
                ]}
            ], "max_tokens": max_tokens, "temperature": temperature, "n": n
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=(30,60))
            response.raise_for_status()
            ans = [choice['message']['content'] for choice in response.json().get('choices', []) if choice.get('message')]
            if not ans: return ['{"latitude": 0.0,"longitude": 0.0}'] * n
            while len(ans) < n: ans.append(ans[0] if ans else '{"latitude": 0.0,"longitude": 0.0}')
            return ans[:n]
        except requests.RequestException as e:
            # print(f"API RAG request failed: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * n
        except Exception as e:
            # print(f"Error processing OpenAI RAG response: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * n

    def _process_row_openai(self, row, root_path, image_path_segment, detail, max_tokens, temperature, n):
        img_full_path = os.path.join(root_path, image_path_segment, row["IMG_ID"])
        try:
            response_list = self.get_openai_response(img_full_path, detail, max_tokens, temperature, n)
        except Exception as e:
            # print(f"Error in _process_row_openai for {img_full_path}: {e}") # Reduce noise
            response_list = ['{"latitude": 0.0,"longitude": 0.0}'] * (n or self.default_n_choices)
        row['response'] = response_list 
        return row

    def _process_row_rag_openai(self, row, root_path, image_path_segment, rag_sample_num, detail, max_tokens, temperature, n):
        img_full_path = os.path.join(root_path, image_path_segment, row["IMG_ID"])
        try:
            candidates_gps_str = str([row[f'candidate_{i}_gps'] for i in range(rag_sample_num)])
            reverse_gps_str = str([row[f'reverse_{i}_gps'] for i in range(rag_sample_num)])
            response_list = self.get_openai_response_rag(img_full_path, candidates_gps_str, reverse_gps_str, detail, max_tokens, temperature, n)
        except Exception as e:
            # print(f"Error in _process_row_rag_openai for {img_full_path}: {e}") # Reduce noise
            response_list = ['{"latitude": 0.0,"longitude": 0.0}'] * (n or self.default_n_choices)
        row['rag_response'] = response_list
        return row

    def run_predictions(self, args):
        if PANDARALLEL_AVAILABLE:
             pandarallel.initialize(progress_bar=TQDM_AVAILABLE, nb_workers=getattr(args, 'num_pandarallel_workers', 4))
        # else:
            # print("Pandarallel is not installed. Operations will run sequentially. This might be slow.") # Reduce noise


        df = pd.read_csv(os.path.join(args.root_path, args.text_path))
        df = df.head(getattr(args, 'nrows_to_process', len(df))) # Process only a subset if specified

        detail = getattr(args, 'detail', None)
        max_tokens = getattr(args, 'max_tokens', None)
        temperature = getattr(args, 'temperature', None)
        n_choices = getattr(args, 'n_choices', None)

        if args.process == 'predict':
            # print("Starting 'predict' process...") # Reduce noise
            if PANDARALLEL_AVAILABLE and df.shape[0] > getattr(args, 'pandarallel_threshold', 100) : 
                df = df.parallel_apply(lambda row: self._process_row_openai(row, args.root_path, args.image_path, detail, max_tokens, temperature, n_choices), axis=1)
            else:
                df_iterator = tqdm(df.iterrows(), total=df.shape[0], desc="Processing predict") if TQDM_AVAILABLE else df.iterrows()
                df = df.apply(lambda row: self._process_row_openai(row, args.root_path, args.image_path, detail, max_tokens, temperature, n_choices), axis=1)
            df.to_csv(os.path.join(args.root_path, args.result_path), index=False)
            # print(f"Predict process finished. Results saved to {args.result_path}") # Reduce noise

        elif args.process == 'extract':
            # print("Starting 'extract' process...") # Reduce noise
            df_results = pd.read_csv(os.path.join(args.root_path, args.result_path))
            
            def extract_coords_from_response_list(response_list_str):
                try:
                    actual_list_of_json_strings = ast.literal_eval(response_list_str)
                    parsed_coords_list = []
                    for json_str in actual_list_of_json_strings:
                        parsed_coords_list.append(_parse_llm_response_to_coords(json_str))
                    return parsed_coords_list
                except (ValueError, SyntaxError, TypeError):
                    return [[0.0, 0.0]] * (n_choices or self.default_n_choices) # Ensure consistent list length

            df_results['coordinates'] = df_results['response'].apply(extract_coords_from_response_list)
            df_results.to_csv(os.path.join(args.root_path, args.result_path), index=False)
            # print(f"Extract process finished. Results updated in {args.result_path}") # Reduce noise
            
        elif args.process == 'rag':
            # print("Starting 'rag' process...") # Reduce noise
            database_df = pd.read_csv(getattr(args, 'mp16_database_path', './data/MP16_Pro_filtered.csv')) 
            output_rag_path = os.path.join(args.root_path, str(args.rag_sample_num) + '_' + args.rag_path)

            if not os.path.exists(output_rag_path) or getattr(args, 'force_rag_preprocess', False):
                # print("Preprocessing for RAG: Loading search indices and assigning candidate/reverse GPS points...") # Reduce noise
                I = np.load(os.path.join('./index/', f'{args.searching_file_name}.npy'))
                reverse_I = np.load(os.path.join('./index/', f'{args.searching_file_name}_reverse.npy'))
                
                df_iterator_rag_prep = tqdm(range(df.shape[0]), desc="Assigning RAG candidates") if TQDM_AVAILABLE else range(df.shape[0])
                for i in df_iterator_rag_prep:
                    candidate_indices = I[i][:args.rag_sample_num] 
                    reverse_indices = reverse_I[i][:args.rag_sample_num]
                    valid_candidate_indices = [idx for idx in candidate_indices if idx < len(database_df)]
                    valid_reverse_indices = [idx for idx in reverse_indices if idx < len(database_df)]
                    candidate_gps_data = database_df.loc[valid_candidate_indices, ['LAT', 'LON']].values.tolist()
                    reverse_gps_data = database_df.loc[valid_reverse_indices, ['LAT', 'LON']].values.tolist()
                    while len(candidate_gps_data) < args.rag_sample_num: candidate_gps_data.append([0.0,0.0])
                    while len(reverse_gps_data) < args.rag_sample_num: reverse_gps_data.append([0.0,0.0])
                    for idx_rag in range(args.rag_sample_num):
                        df.loc[i, f'candidate_{idx_rag}_gps'] = str(candidate_gps_data[idx_rag])
                        df.loc[i, f'reverse_{idx_rag}_gps'] = str(reverse_gps_data[idx_rag])
                df.to_csv(output_rag_path, index=False) 
                # print(f"RAG preprocessing complete. Data saved to {output_rag_path}") # Reduce noise
            else:
                # print(f"Found existing RAG preprocessed file: {output_rag_path}. Loading it.") # Reduce noise
                df = pd.read_csv(output_rag_path)
                df = df.head(getattr(args, 'nrows_to_process', len(df))) # Re-apply nrows_to_process if loading existing

            if PANDARALLEL_AVAILABLE and df.shape[0] > getattr(args, 'pandarallel_threshold', 100):
                df = df.parallel_apply(lambda row: self._process_row_rag_openai(row, args.root_path, args.image_path, args.rag_sample_num, detail, max_tokens, temperature, n_choices), axis=1)
            else:
                df_iterator_rag = tqdm(df.iterrows(), total=df.shape[0], desc="Processing RAG") if TQDM_AVAILABLE else df.iterrows()
                df = df.apply(lambda row: self._process_row_rag_openai(row, args.root_path, args.image_path, args.rag_sample_num, detail, max_tokens, temperature, n_choices), axis=1)
            df.to_csv(output_rag_path, index=False)
            # print(f"RAG process finished. Results saved to {output_rag_path}") # Reduce noise

        elif args.process == 'rag_extract':
            # print("Starting 'rag_extract' process...") # Reduce noise
            rag_file_path = os.path.join(args.root_path, str(getattr(args, 'rag_sample_num', 5)) + '_' + args.rag_path) 
            if not os.path.exists(rag_file_path):
                 rag_file_path = os.path.join(args.root_path, args.rag_path)
                 # print(f"Trying alternative RAG file path: {rag_file_path}") # Reduce noise

            df_results = pd.read_csv(rag_file_path).fillna("None")
            
            def extract_coords_from_rag_response_list(response_list_str):
                try:
                    actual_list_of_json_strings = ast.literal_eval(response_list_str)
                    parsed_coords_list = []
                    for json_str in actual_list_of_json_strings:
                        parsed_coords_list.append(_parse_llm_response_to_coords(json_str))
                    return parsed_coords_list
                except (ValueError, SyntaxError, TypeError):
                    return [[0.0, 0.0]] * (n_choices or self.default_n_choices) 

            df_results['rag_coordinates'] = df_results['rag_response'].apply(extract_coords_from_rag_response_list)
            df_results.to_csv(rag_file_path, index=False)
            # print(f"RAG extract process finished. Results updated in {rag_file_path}") # Reduce noise
        else:
            print(f"Unknown process type: {args.process}")


class HuggingFaceLlavaGeolocator:
    def __init__(self, model_path, torch_dtype_str='float16', device_map="auto"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers or PyTorch is not installed. HuggingFaceLlavaGeolocator cannot be used.")
        
        self.model_path = model_path
        try:
            self.torch_dtype = getattr(torch, torch_dtype_str)
        except AttributeError:
            print(f"Warning: torch_dtype '{torch_dtype_str}' not recognized. Defaulting to torch.float16.")
            self.torch_dtype = torch.float16
        self.device_map = device_map
        
        # print(f"Initializing HuggingFaceLlavaGeolocator with model: {model_path}") # Reduce noise
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=self.torch_dtype, 
            device_map=self.device_map
        )
        # print("Llava model and processor loaded.") # Reduce noise
        self.default_max_tokens = 200 
        self.default_temperature = 0.7 
        self.default_n_choices = 10 # Llava generate might not support num_return_sequences directly like OpenAI. It's often 1.

    def get_hf_response(self, image_path, max_tokens=None, temperature=None, n=None):
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        # 'n' (num_return_sequences) needs careful handling for HF model.generate.
        # For simplicity, let's assume n=1 for HF model unless explicitly supported and tested.
        num_return_sequences = n if n is not None and n > 0 else 1 # Default to 1 if n is problematic
        if n is not None and n > 1:
             print(f"Warning: n_choices={n} for HuggingFace model. Ensure model.generate supports num_return_sequences > 1. Defaulting to 1 if issues arise.")
             # Some models might need specific setup for num_return_sequences > 1 (e.g. beam search)

        try:
            image = Image.open(image_path).convert('RGB') # Ensure RGB
        except Exception as e:
            # print(f"Error opening image {image_path}: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * num_return_sequences

        conversation = [{"role": "user", "content": [
            {"type": "text", "text": """Suppose you are an expert in geo-localization, you have the ability to give two number GPS coordination given an image.
            Please give me the location of the given image.
            Remember, you must have an answer, just output your best guess, don't answer me that you can't give a location.
            Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}.
            Your answer should be in the following JSON format without any other information: {"latitude": float,"longitude": float}."""},
            {"type": "image"},
        ]}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)
        
        try:
            pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, num_return_sequences=num_return_sequences, do_sample=True, pad_token_id=pad_token_id)
            dialogue = self.processor.batch_decode(output, skip_special_tokens=True)
            ans = [d.split("assistant")[-1].strip() for d in dialogue]
            if not ans: return ['{"latitude": 0.0,"longitude": 0.0}'] * num_return_sequences
            while len(ans) < num_return_sequences: ans.append(ans[0] if ans else '{"latitude": 0.0,"longitude": 0.0}')
            return ans[:num_return_sequences]
        except Exception as e:
            # print(f"Error during HF model generation: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * num_return_sequences

    def get_hf_response_rag(self, image_path, candidates_gps, reverse_gps, max_tokens=None, temperature=None, n=None):
        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature or self.default_temperature
        num_return_sequences = n if n is not None and n > 0 else 1
        if n is not None and n > 1:
             print(f"Warning: n_choices={n} for HuggingFace RAG. Ensure model.generate supports num_return_sequences > 1.")


        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # print(f"Error opening image {image_path}: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * num_return_sequences

        conversation = [{"role": "user", "content": [
            {"type": "text", "text": f"""Suppose you are an expert in geo-localization, Please analyze this image and give me a guess of the location.
                Your answer must be to the coordinates level in (latitude, longitude) format.
                For your reference, these are coordinates of some similar images: {candidates_gps}, and these are coordinates of some dissimilar images: {reverse_gps}.
                Remember, you must have an answer, just output your best guess, don't answer me that you can't give an location.
                Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}.
                Your answer should be in the following JSON format without any other information: {{"latitude": float,"longitude": float}}."""},
            {"type": "image"},
        ]}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.model.device)

        try:
            pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, num_return_sequences=num_return_sequences, do_sample=True, pad_token_id=pad_token_id)
            dialogue = self.processor.batch_decode(output, skip_special_tokens=True)
            ans = [d.split("assistant")[-1].strip() for d in dialogue]
            if not ans: return ['{"latitude": 0.0,"longitude": 0.0}'] * num_return_sequences
            while len(ans) < num_return_sequences: ans.append(ans[0] if ans else '{"latitude": 0.0,"longitude": 0.0}')
            return ans[:num_return_sequences]
        except Exception as e:
            # print(f"Error during HF RAG model generation: {e}") # Reduce noise
            return ['{"latitude": 0.0,"longitude": 0.0}'] * num_return_sequences

    def _process_row_hf(self, row, root_path, image_path_segment, max_tokens, temperature, n):
        img_full_path = os.path.join(root_path, image_path_segment, row["IMG_ID"])
        try:
            response_list = self.get_hf_response(img_full_path, max_tokens, temperature, n)
        except Exception as e:
            # print(f"Error in _process_row_hf for {img_full_path}: {e}") # Reduce noise
            response_list = ['{"latitude": 0.0,"longitude": 0.0}'] * (n or 1) # Default n to 1 for HF
        row['response'] = response_list
        return row

    def _process_row_rag_hf(self, row, root_path, image_path_segment, rag_sample_num, max_tokens, temperature, n):
        img_full_path = os.path.join(root_path, image_path_segment, row["IMG_ID"])
        try:
            candidates_gps_str = str([row[f'candidate_{i}_gps'] for i in range(rag_sample_num)])
            reverse_gps_str = str([row[f'reverse_{i}_gps'] for i in range(rag_sample_num)])
            response_list = self.get_hf_response_rag(img_full_path, candidates_gps_str, reverse_gps_str, max_tokens, temperature, n)
        except Exception as e:
            # print(f"Error in _process_row_rag_hf for {img_full_path}: {e}") # Reduce noise
            response_list = ['{"latitude": 0.0,"longitude": 0.0}'] * (n or 1)
        row['rag_response'] = response_list
        return row

    def run_predictions(self, args):
        if TQDM_AVAILABLE:
            tqdm.pandas(desc="Processing rows with HF LLaVA")
        # else:
            # print("tqdm not available, pandas operations will not show progress bars.") # Reduce noise

        df = pd.read_csv(os.path.join(args.root_path, args.text_path))
        df = df.head(getattr(args, 'nrows_to_process', len(df)))

        max_tokens = getattr(args, 'max_tokens', None)
        temperature = getattr(args, 'temperature', None)
        n_choices = getattr(args, 'n_choices', 1) # Default n_choices to 1 for HF as multiple returns are less common

        if args.process == 'predict':
            # print("Starting HF 'predict' process...") # Reduce noise
            if TQDM_AVAILABLE and df.shape[0] > getattr(args, 'tqdm_threshold', 100) : # Use tqdm for larger dataframes
                df = df.progress_apply(lambda row: self._process_row_hf(row, args.root_path, args.image_path, max_tokens, temperature, n_choices), axis=1)
            else: # Simple apply for smaller ones or if tqdm not available
                df = df.apply(lambda row: self._process_row_hf(row, args.root_path, args.image_path, max_tokens, temperature, n_choices), axis=1)
            df.to_csv(os.path.join(args.root_path, args.result_path), index=False)
            # print(f"HF Predict process finished. Results saved to {args.result_path}") # Reduce noise

        elif args.process == 'extract':
            # print("Starting HF 'extract' process...") # Reduce noise
            df_results = pd.read_csv(os.path.join(args.root_path, args.result_path))
            def extract_coords_from_response_list_hf(response_list_str):
                try:
                    actual_list_of_json_strings = ast.literal_eval(response_list_str)
                    parsed_coords_list = []
                    for json_str in actual_list_of_json_strings:
                        parsed_coords_list.append(_parse_llm_response_to_coords(json_str))
                    return parsed_coords_list
                except (ValueError, SyntaxError, TypeError):
                    return [[0.0, 0.0]] * (n_choices or 1)
            df_results['coordinates'] = df_results['response'].apply(extract_coords_from_response_list_hf)
            df_results.to_csv(os.path.join(args.root_path, args.result_path), index=False)
            # print(f"HF Extract process finished. Results updated in {args.result_path}") # Reduce noise

        elif args.process == 'rag':
            # print("Starting HF 'rag' process...") # Reduce noise
            database_df = pd.read_csv(getattr(args, 'mp16_database_path', './data/MP16_Pro_filtered.csv'))
            output_rag_path = os.path.join(args.root_path, str(args.rag_sample_num) + '_' + args.rag_path)

            if not os.path.exists(output_rag_path) or getattr(args, 'force_rag_preprocess', False):
                # print("Preprocessing for HF RAG: Loading search indices and assigning candidate/reverse GPS points...") # Reduce noise
                I = np.load(os.path.join('./index/', f'{args.searching_file_name}.npy'))
                reverse_I = np.load(os.path.join('./index/', f'{args.searching_file_name}_reverse.npy'))
                
                df_iterator_hf_rag_prep = tqdm(range(df.shape[0]), desc="Assigning RAG candidates for HF") if TQDM_AVAILABLE else range(df.shape[0])
                for i in df_iterator_hf_rag_prep:
                    candidate_indices = I[i][:args.rag_sample_num]
                    reverse_indices = reverse_I[i][:args.rag_sample_num]
                    valid_candidate_indices = [idx for idx in candidate_indices if idx < len(database_df)]
                    valid_reverse_indices = [idx for idx in reverse_indices if idx < len(database_df)]
                    candidate_gps_data = database_df.loc[valid_candidate_indices, ['LAT', 'LON']].values.tolist()
                    reverse_gps_data = database_df.loc[valid_reverse_indices, ['LAT', 'LON']].values.tolist()
                    while len(candidate_gps_data) < args.rag_sample_num: candidate_gps_data.append([0.0,0.0])
                    while len(reverse_gps_data) < args.rag_sample_num: reverse_gps_data.append([0.0,0.0])
                    for idx_rag in range(args.rag_sample_num):
                        df.loc[i, f'candidate_{idx_rag}_gps'] = str(candidate_gps_data[idx_rag])
                        df.loc[i, f'reverse_{idx_rag}_gps'] = str(reverse_gps_data[idx_rag])
                df.to_csv(output_rag_path, index=False)
                # print(f"HF RAG preprocessing complete. Data saved to {output_rag_path}") # Reduce noise
            else:
                # print(f"Found existing HF RAG preprocessed file: {output_rag_path}. Loading it.") # Reduce noise
                df = pd.read_csv(output_rag_path)
                df = df.head(getattr(args, 'nrows_to_process', len(df)))


            if TQDM_AVAILABLE and df.shape[0] > getattr(args, 'tqdm_threshold', 100):
                df = df.progress_apply(lambda row: self._process_row_rag_hf(row, args.root_path, args.image_path, args.rag_sample_num, max_tokens, temperature, n_choices), axis=1)
            else:
                df = df.apply(lambda row: self._process_row_rag_hf(row, args.root_path, args.image_path, args.rag_sample_num, max_tokens, temperature, n_choices), axis=1)
            df.to_csv(output_rag_path, index=False)
            # print(f"HF RAG process finished. Results saved to {output_rag_path}") # Reduce noise

        elif args.process == 'rag_extract':
            # print("Starting HF 'rag_extract' process...") # Reduce noise
            rag_file_path = os.path.join(args.root_path, str(getattr(args, 'rag_sample_num', 5)) + '_' + args.rag_path)
            if not os.path.exists(rag_file_path):
                 rag_file_path = os.path.join(args.root_path, args.rag_path)
                 # print(f"Trying alternative RAG file path: {rag_file_path}") # Reduce noise

            df_results = pd.read_csv(rag_file_path).fillna("None")
            def extract_coords_from_rag_response_list_hf(response_list_str):
                try:
                    actual_list_of_json_strings = ast.literal_eval(response_list_str)
                    parsed_coords_list = []
                    for json_str in actual_list_of_json_strings:
                        parsed_coords_list.append(_parse_llm_response_to_coords(json_str))
                    return parsed_coords_list
                except (ValueError, SyntaxError, TypeError):
                    return [[0.0, 0.0]] * (n_choices or 1)
            df_results['rag_coordinates'] = df_results['rag_response'].apply(extract_coords_from_rag_response_list_hf)
            df_results.to_csv(rag_file_path, index=False)
            # print(f"HF RAG extract process finished. Results updated in {rag_file_path}") # Reduce noise
        else:
            print(f"Unknown process type: {args.process}")
