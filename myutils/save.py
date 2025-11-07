import torch
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def safe_load_model(file_path: str, device: Optional[str] = None, 
                    custom_objects: Optional[Dict] = None
                    ) -> Any:
    """
    Safely load PyTorch model, handling serialization warnings
    
    Args:
        file_path: Model file path
        device: Device to load the model on
        custom_objects: Custom objects dictionary for safe loading
    
    Returns:
        Loaded model
    """
    if custom_objects:
        # Provide safe loading support for custom objects
        torch.serialization.add_safe_globals(custom_objects)
        
    # Use weights_only=True for safe loading
    try:
        if device:
            return torch.load(file_path, 
                              map_location=device, 
                              )
        else:
            return torch.load(file_path)
    except TypeError:
        # Compatible with older PyTorch versions
        if device:
            return torch.load(file_path, 
                              map_location=device
                              )
        else:
            return torch.load(file_path)