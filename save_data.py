from myutils.save import safe_load_model
import torch
import numpy as np
import yaml
import os
from typing import Dict, Tuple, Any, List

def save_dataset(files: Dict) -> None:
    """
    Save dataset
    
    Args:
        files: Data file configuration
    """
    from load_data import load_data
    from process_data import process
    drug_feature, mutation_feature, \
    gexpr_feature, methylation_feature, \
    data_new, num_celllines, num_drugs = load_data(files)
    drug_set, cellline_set, \
    allpairs, atom_shape = process(
        drug_feature, mutation_feature, \
        gexpr_feature, methylation_feature, 
        data_new, num_celllines, num_drugs
    )
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(files['drug']['tensor']), exist_ok=True)
    os.makedirs(os.path.dirname(files['cell_line']['tensor']), exist_ok=True)
    os.makedirs(os.path.dirname(files['allpairs']), exist_ok=True)
    
    # Use safe saving method
    try:
        torch.save(drug_set, files['drug']['tensor'], _use_new_zipfile_serialization=True)
        torch.save(cellline_set, files['cell_line']['tensor'], _use_new_zipfile_serialization=True)
    except TypeError:
        # Compatible with newest PyTorch version
        torch.save(drug_set, files['drug']['tensor'])
        torch.save(cellline_set, files['cell_line']['tensor'])
    
    np.save(files['allpairs'], allpairs)
    print(f"Dataset saved successfully, atom feature dimension: {atom_shape}")


def load_dataset(files: Dict) -> Tuple[Any, Any, np.ndarray, int]:
    """
    Load dataset
    
    Args:
        files: Data file configuration
    
    Returns:
        drug_set: Drug data loader
        cellline_set: Cell line data loader
        allpairs: All drug-cell line pairs
        atom_shape: Atom feature shape
    """
    # Use custom classes as safe loading options
    custom_objects = {
        "GraphDataset": __import__("myutils.dataset", fromlist=["GraphDataset"]).GraphDataset,
        "collate": __import__("myutils.process", fromlist=["collate"]).collate
    }
    
    # Use safe loading method
    drug_set = safe_load_model(files['drug']['tensor'], custom_objects=custom_objects)
    cellline_set = safe_load_model(files['cell_line']['tensor'], custom_objects=custom_objects)
    allpairs = np.load(files['allpairs'], allow_pickle=True)
    
    return drug_set, cellline_set, allpairs

if __name__ == '__main__':
    try:
        # Try loading the configuration file
        with open('config.yaml', 'r', encoding='utf-8') as yaml_file:
            config = yaml.safe_load(yaml_file)
        
        if 'data' not in config:
            print("Error: 'data' field not found in configuration file")
            exit(1)
            
        arg_files = config['data']
        
        # Process each dataset
        for idx, files in enumerate(arg_files):
            print(f"Processing dataset {idx+1}/{len(arg_files)}: {files.get('name', 'unknown')}")
            try:
                save_dataset(files)
                print(f"Dataset {idx+1} processing completed")
            except Exception as e:
                print(f"Error processing dataset {idx+1}: {e}")
    
    except FileNotFoundError:
        print("Error: Configuration file 'config.yaml' not found")
    except yaml.YAMLError:
        print("Error: Configuration file format is incorrect")
    except Exception as e:
        print(f"An unknown error occurred: {e}")