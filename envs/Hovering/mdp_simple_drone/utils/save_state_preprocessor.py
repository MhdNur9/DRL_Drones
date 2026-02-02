import yaml
import torch
import os

def save_preprocessor_state(path_to_model: str, save_path: str):
    """
    Save the state preprocessor from a trained model to a YAML file.
    This function extracts the state preprocessor from a saved PyTorch model and
    saves it as a human-readable YAML file. Tensor values are converted to lists
    for better readability in the YAML format.
    Args:
        path_to_model (str): Path to the saved .pt PyTorch model file containing
                            the state preprocessor in the model dictionary.
        save_path (str): Directory path where the state preprocessing YAML file
                        will be saved. The file will be named 'state_preprocessing.yaml'.
    Returns:
        None
    Raises:
        FileNotFoundError: If the model file at path_to_model doesn't exist.
        KeyError: If the model dictionary doesn't contain 'state_preprocessor' key.
        OSError: If there are issues writing to the save_path directory.
    Example:
        >>> save_preprocessor_state('/path/to/model.pth', '/path/to/output/dir')
        # Creates '/path/to/output/dir/state_preprocessing.yaml'
    """
    model_dict = torch.load(path_to_model, map_location=torch.device("cpu"))
    state_preprocessor = model_dict["state_preprocessor"]

    # Convert tensors -> lists for YAML readability
    clean_state = {}
    for k, v in state_preprocessor.items():
        if torch.is_tensor(v):
            clean_state[k] = v.tolist()
        else:
            clean_state[k] = v

    with open(os.path.join(save_path, "state_preprocessing.yaml"), "w") as f:
        yaml.safe_dump(clean_state, f, sort_keys=False, default_flow_style=True)
