import numpy as np
import csv
import pandas as pd
import os
import pickle
from typing import Dict, Tuple
from pathlib import Path

from rdkit import Chem
import deepchem as dc

from myutils.dataloader.GDSC import DataLoader as GDSCDataLoader
from myutils.dataloader.CCLE import DataLoader as CCLEDataLoader

def load_data(files: Dict) -> Tuple:
    """
    Load dataset
    
    Args:
        files: Data file configuration dictionary
        
    Returns:
        Tuple of drug features, mutation features, gene expression features, methylation features, data, number of cell lines, number of drugs
        
    Raises:
        NotImplementedError: If the dataset name is not supported
        FileNotFoundError: If the file does not exist
    """
    # Check if files exist
    if files['name'] == 'GDSC':
        paths = [
            files['drug']['info'], 
            files['drug']['features'], 
            files['cell_line']['info'], 
            files['cell_line']['genomic']['mutation'],
            files['cell_line']['genomic']['expression'], 
            files['cell_line']['genomic']['methylation'],
            files['cell_line']['response'], 
            files['drug']['thresholds']
        ]
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"File does not exist: {path}")
                
        loader = GDSCDataLoader(*paths)
        return loader()
    elif files['name'] == 'CCLE':
        paths = [
            files['drug']['features'], 
            files['genomic']['mutation'], 
            files['genomic']['expression'], 
            files['genomic']['methylation'], 
            files['cancer']['response']
        ]
        for path in paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"File does not exist: {path}")
                
        loader = CCLEDataLoader(*paths)
        return loader()
    else:
        raise NotImplementedError(f"Unsupported dataset name: "
                                  f"{files['name']}"
                                  )
