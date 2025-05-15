import os
import pickle

def save_cache(object: any, save_path: str): 
  """
    Save a Python object to disk at the specified path.

    Creates necessary directories if they don't exist and serializes the object using pickle.

    Args:
        save_path (str): File path where the object should be saved.
        object: Any Python object that can be pickled.
    """
  os.makedirs(os.path.dirname(save_path), exist_ok=True)

  if save_path is not None: 
    with open(save_path, "wb") as f: 
      pickle.dump(obj=object, file=f)
  
  else: 
    raise ValueError(f"Path does not exit: {save_path}")
  

def read_cache(cache_path: str): 
  """
  Read a previously saved Python object from disk if available.

  Args:
      cache_path (str): File path where the object was saved.

  Returns:
      data: The loaded Python object if successful, None otherwise.
  """
  if os.path.exists(cache_path): 
    with open(cache_path, 'rb') as f: 
      data = pickle.load(f)
      return data
  else:
    raise ValueError(f"Path does not exit: {cache_path}")