import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from myutils.dataset import GraphDataset
from myutils.process import collate
from scipy.sparse import coo_matrix
from typing import Tuple, List, Dict


def CalculateGraphFeat(feat_mat: np.ndarray, adj_list: List) -> List:
    """
    Calculate graph features
    
    Args:
        feat_mat: Feature matrix
        adj_list: Adjacency list
    
    Returns:
        List containing feature matrix and edge indices
    """
    assert feat_mat.shape[0] == len(adj_list), "Feature matrix rows must equal adjacency list length"
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T), \
        "Adjacency matrix must be symmetric"
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def FeatureExtract(drug_feature: pd.DataFrame) -> List:
    """
    Extract drug features
    
    Args:
        drug_feature: Drug feature DataFrame
    
    Returns:
        Extracted feature list
    """
    drug_data = [[] for _ in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat, adj_list, _ = drug_feature.iloc[i]
        drug_data[i] = CalculateGraphFeat(feat_mat, adj_list)
    return drug_data


# Transform the data into inputs
def process(drug_feature: Dict, 
           mutation_feature: pd.DataFrame, 
           gexpr_feature: pd.DataFrame, 
           methylation_feature: pd.DataFrame, 
           data_new: List, 
           num_celllines: int, 
           num_drugs: int) -> Tuple:
    """
    Process data into model input format
    
    Args:
        drug_feature: Drug features
        mutation_feature: Mutation features
        gexpr_feature: Gene expression features
        methylation_feature: Methylation features
        data_new: Drug-cell line pair data
        num_celllines: Number of cell lines
        num_drugs: Number of drugs
    
    Returns:
        drug_set: Drug data loader
        cellline_set: Cell line data loader
        allpairs: All drug-cell line pairs
        atom_shape: Atom feature shape
    """
    # construct cell line-drug response pairs
    cellineid = list(set([item[0] for item in data_new]))
    cellineid.sort()
    cid = list(set([item[1] for item in data_new]))
    cid.sort()

    cellmap = list(zip(cellineid, list(range(len(cellineid)))))
    np.save('cellmap.npy', cellmap)
    pubmedmap = list(zip(cid, list(range(len(cellineid), len(cellineid) + len(cid)))))
    np.save('pubmedmap.npy', pubmedmap)
    cellline_num = np.squeeze([[j[1] for j in cellmap if i[0] == j[0]] for i in data_new])
    pubmed_num = np.squeeze([[j[1] for j in pubmedmap if i[1] == j[0]] for i in data_new])
    IC_num = np.squeeze([i[2] for i in data_new])
    allpairs = np.vstack((cellline_num, pubmed_num, IC_num)).T
    allpairs = allpairs[allpairs[:, 2].argsort()]

    # process drug feature
    pubid = [item[0] for item in pubmedmap]
    drug_feature_df = pd.DataFrame(drug_feature).T
    drug_feature_df = drug_feature_df.loc[pubid]
    atom_shape = drug_feature_df[0][0].shape[-1]
    drug_data = FeatureExtract(drug_feature_df)

    # process cell line feature input
    cellid = [item[0] for item in cellmap]
    gexpr_feature = gexpr_feature.loc[cellid]
    mutation_feature = mutation_feature.loc[cellid]
    methylation_feature = methylation_feature.loc[cellid]

    # Convert to PyTorch tensors
    mutation = torch.from_numpy(np.array(mutation_feature, dtype='float32'))
    mutation = torch.unsqueeze(mutation, dim=1)
    mutation = torch.unsqueeze(mutation, dim=1)
    gexpr = torch.from_numpy(np.array(gexpr_feature, 
                                      dtype='float32'
                                      ))
    methylation = torch.from_numpy(np.array(methylation_feature, 
                                            dtype='float32'
                                            ))

    # Create data loaders
    drug_set = Data.DataLoader(
        dataset=GraphDataset(graphs_dict=drug_data), 
        collate_fn=collate, 
        batch_size=num_drugs, 
        shuffle=False, 
        num_workers=0
    )
    
    cellline_set = Data.DataLoader(
        dataset=Data.TensorDataset(mutation, gexpr, methylation), 
        batch_size=num_celllines, 
        shuffle=False, 
        num_workers=0
    )

    # Save datasets
    try:
        torch.save(drug_set, 'drug_set.pt', _use_new_zipfile_serialization=True)
        torch.save(cellline_set, 'cellline_set.pt', _use_new_zipfile_serialization=True)
    except TypeError:
        # Compatible with newest PyTorch version
        torch.save(drug_set, 'drug_set.pt')
        torch.save(cellline_set, 'cellline_set.pt')
    
    return drug_set, cellline_set, allpairs, atom_shape

class Sampler(object):
    """Data sampler for creating training, validation and test sets"""
    def __init__(self, allpairs: np.ndarray, num_celllines: int, num_drugs: int, 
                train_ratio: float, valid_ratio: float, seed: int):
        """
        Initialize sampler
        
        Args:
            allpairs: All drug-cell line pair data
            num_celllines: Number of cell lines
            num_drugs: Number of drugs
            train_ratio: Training set ratio
            valid_ratio: Validation set ratio
            seed: Random seed
        """
        train_mask, valid_mask, test_mask = self.cmask(len(allpairs), train_ratio, valid_ratio, seed)
        self.train_mask = self.make_mask(train_mask, allpairs, num_celllines, num_drugs)
        self.valid_mask = self.make_mask(valid_mask, allpairs, num_celllines, num_drugs)
        self.test_mask = self.make_mask(test_mask, allpairs, num_celllines, num_drugs)
        self.train_edge = np.vstack([allpairs[train_mask][:, 0:3], allpairs[train_mask][:, 0:3][:, [1, 0, 2]]])
        self.label_pos = self.make_pos_label(allpairs, num_celllines, num_drugs)

    @staticmethod
    def cmask(num: int, train_ratio: float, valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create masks for training, validation and test sets
        
        Args:
            num: Total number of data points
            train_ratio: Training set ratio
            valid_ratio: Validation set ratio
            seed: Random seed
            
        Returns:
            train_mask: Training set mask
            valid_mask: Validation set mask
            test_mask: Test set mask
        """
        mask = np.zeros(num)
        mask[0:int(train_ratio * num)] = 0
        mask[int(train_ratio * num):int((train_ratio+valid_ratio) * num)] = 1
        mask[int((train_ratio + valid_ratio) * num):] = 2
        np.random.seed(seed)
        np.random.shuffle(mask)
        train_mask = (mask == 0)
        valid_mask = (mask == 1)
        test_mask = (mask == 2)
        return train_mask, valid_mask, test_mask
    
    @staticmethod
    def make_mask(mask: np.ndarray, allpairs: np.ndarray, num_celllines: int, num_drugs: int) -> torch.Tensor:
        """
        Create mask tensor
        
        Args:
            mask: Boolean mask array
            allpairs: All drug-cell line pair data
            num_celllines: Number of cell lines
            num_drugs: Number of drugs
            
        Returns:
            Mask tensor
        """
        index = allpairs[mask][:, 0:3]
        matrix = coo_matrix(
            (np.ones(index.shape[0], dtype=bool), (index[:, 0], index[:, 1])),
            shape=(num_celllines, num_drugs)
        ).toarray()
        return torch.from_numpy(matrix).view(-1)
    
    @staticmethod
    def make_pos_label(allpairs: np.ndarray, num_celllines: int, num_drugs: int) -> torch.Tensor:
        """
        Create positive sample labels
        
        Args:
            allpairs: All drug-cell line pair data
            num_celllines: Number of cell lines
            num_drugs: Number of drugs
            
        Returns:
            Positive sample label tensor
        """
        pos_pairs = allpairs[allpairs[:, 2] == 1]
        pos_pairs[:, 1] -= num_celllines
        label_pos = coo_matrix(
            (np.ones(pos_pairs.shape[0]), (pos_pairs[:, 0], pos_pairs[:, 1])),
            shape=(num_celllines, num_drugs)
        ).toarray()
        return torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)