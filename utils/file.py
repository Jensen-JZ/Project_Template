import glob
import json
import os
import pickle
import shutil


def list_all_images(path, full_path=True, sort=True):
    """
    Recursively lists all image files in the given directory.

    Args:
        path (str): Root directory to search for images.
        full_path (bool): If True, return full paths. If False, return paths relative to `path`.
        sort (bool): If True, sort the list of image paths.
        
    Returns:
        List[str]: List of image file paths with consistent forward slashes.
    """

    image_types = [
        '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff',
        '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF'
    ]
    image_path_list = []
    for image_type in image_types:
        image_path_list.extend(glob.glob(os.path.join(path, f'**/*.{image_type}'), recursive=True))
    if not full_path:
        image_path_list = [os.path.relpath(image_path, path) for image_path in image_path_list]
    image_path_list = [image_path.replace('\\', '/') for image_path in image_path_list]
    if sort:
        image_path_list.sort()
    return image_path_list


def list_sub_folders(path, full_path=True, sort=True):
    """
    Lists all subfolders in the given directory.

    Args:
        path (str): Root directory to search for subfolders.
        full_path (bool): If True, return full paths. If False, return only folder names.
        sort (bool): If True, sort the list of subfolder paths.

    Returns:
        List[str]: List of subfolder paths or names.
    """
    
    folders_list = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folder_path = os.path.join(path, folder) if full_path else folder
            folders_list.append(folder_path)
    folders_list = [folder.replace('\\', '/') for folder in folders_list]
    if sort:
        folders_list.sort()
    return folders_list


def make_path(path_or_paths):
    """
    Creates directories if they do not exist.

    Args:
        path_or_paths (str or List[str]): A path or list of paths to create.

    Returns:
        None
    """

    if type(path_or_paths) != list:
        path_or_paths = [path_or_paths]
    for path in path_or_paths:
        os.makedirs(path, exist_ok=True)
        

def prepare_dirs(dirs):
    """
    Ensures that all directories in the list exist.

    Args:
        dirs (list): List of directory paths.

    Returns:
        None
    """

    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)


def save_json(output_dir, config, filename='config'):
    """
    Saves a configuration object to a JSON file.

    Args:
        output_dir (str): Directory where the JSON file will be saved.
        config (object): Configuration object to save (with __dict__).
        filename (str): Name of the JSON file (without extension).

    Returns:
        None
    """
    
    with open(os.path.join(output_dir, f'{filename}.json'), 'w') as f:
        print(json.dumps(config.__dict__, indent=4, sort_keys=True, ensure_ascii=False), file=f)


def write_record(record, file_path, print_to_console=False):
    """
    Appends a string record to a text file and optionally prints it to the console.

    Args:
        record (str): The log or message to write.
        file_path (str): Destination file path.
        print_to_console (bool): Whether to print the record to the terminal.

    Returns:
        None
    """
    
    if print_to_console:
        print(record)
    with open(file_path, 'a') as f:
        f.write(record + '\n')


def delete_dir(path):
    """
    Deletes a directory and all its contents.

    Args:
        path (str): Path to the directory to delete.

    Returns:
        None
    """
    if path is None:
        return
    try:
        shutil.rmtree(path)
    except:
        print(f'Failed to delete {path}. It may not exist or be in use.')


def copy_file(filename, src_dir, dst_dir):
    """
    Copies a single file from source to destination directory.

    Args:
        filename (str): Name of the file to copy.
        src_dir (str): Source directory.
        dst_dir (str): Destination directory.

    Returns:
        None
    """
    shutil.copyfile(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))


def delete_model(model_dir, step):
    """
    Deletes model ckpt files corresponding to a specific training step.

    Args:
        model_dir (str): Directory containing the model ckpt files.
        step (int): Training step number (used in file name pattern).

    Returns:
        None
    """
    
    if step == 0:
        return
    model_files = glob.glob(os.path.join(model_dir, f'{step:06d}*.ckpt'))
    try:
        for model_file in model_files:
            os.remove(model_file)
    except:
        print(f'Failed to delete {model_file}. It may not exist or be in use.')


def get_sample_path(sample_dir, sample_id):
    """
    Constructs the path for a sample file based on its ID.

    Args:
        sample_dir (str): Directory where the sample files are stored.
        sample_id (str or int): Unique ID of the sample.
    
    Returns:
        str: Full path to the sample file.
    """
    return os.path.join(sample_dir, f'sample_{str(sample_id)}')    


def delete_sample(sample_dir, eval_id):
    """
    Deletes a saved sample directory based on its ID.

    Args:
        sample_dir (str): Directory where the sample files are stored.
        eval_id (str or int): Unique ID of the sample to delete.
    
    Returns:
        None
    """

    if not eval_id:
        return
    sample_path = get_sample_path(sample_dir, eval_id)
    try:
        shutil.rmtree(sample_path)
    except:
        print(f'Failed to delete {sample_path}. It may not exist or be in use.')

cache_dir = 'archive/cache'

def save_cache(data, name):
    """
    Saves a Python object to a pickle file in the cache directory.

    Args:
        data (Any): The Python object to serialize.
        name (str): Name of the file to save the object as (without extension).

    Returns:
        None
    """
    
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with open(os.path.join(cache_dir, name), 'wb') as f:
            pickle.dump(data, f)
    except:
        print(f'Failed to save cache {name}. It may not exist or be in use.')


def load_cache(name):
    """
    Loads a Python object from a pickle file in the cache directory.

    Args:
        name (str): Name of the file to load the object from (without extension).

    Returns:
        Any: The deserialized Python object.
    """
    
    with open(os.path.join(cache_dir, name), 'rb') as f:
        return pickle.load(f)
    

def exist_cache(name):
    """
    Checks if a cache file exists in the cache directory.

    Args:
        name (str): Name of the file to check for (without extension).

    Returns:
        bool: True if the cache file exists, False otherwise.
    """
    return os.path.exists(os.path.join(cache_dir, name))


def safe_filename(unsafe, mark=''):
    """
    Converts a potentially unsafe filename into a safe format.

    Replaces slashes, backslashes, and colons with underscores, and appends a mark if provided.

    Args:
        unsafe (str): The original filename (potentially filename) or string to sanitize.
        mark (str): Optional prefix to prepend to the sanitized filename.

    Returns:
        str: A sanitized version of the filename, safe for use in file systems.
    """
    
    if mark:
        unsafe = mark + "__" + unsafe
    unsafe = unsafe.replace('/', '_').replace('\\', '_').replace(':', '_')
    return unsafe