import numpy as np
import torch
from scipy.sparse import coo_matrix


class Sampler(object):
    """
    Data sampler for cross-validation
    Used for 5-fold cross-validation experiments, 
    creating corresponding data masks and training edges 
    based on train_indices and test_indices
    """
    def __init__(self, allpairs, num_celllines, 
                 num_drugs, train_indices, test_indices
                 ):
        """
        Initialize the sampler
        
        Args:
            allpairs: cell line-drug pair data [cell_idx, drug_idx, label]
            num_celllines: number of cell lines
            num_drugs: number of drugs  
            train_indices: training set indices
            test_indices: test set indices
        """
        self.num_celllines = num_celllines
        self.num_drugs = num_drugs
        
        # create training and test masks
        self.train_mask = \
            self._make_mask(train_indices, allpairs, 
                           num_celllines, num_drugs
                           )
        self.test_mask = \
            self._make_mask(test_indices, allpairs, 
                           num_celllines, num_drugs
                           )
        
        # create training edges (keep original indices, 
        # model expects a unified index space)
        train_pairs = allpairs[train_indices]
        self.train_edge = \
            np.vstack([train_pairs[:, 0:3], 
                       train_pairs[:, 0:3][:, [1, 0, 2]]]
                      )

        # create positive sample labels
        self.label_pos = \
            self._make_pos_lable(allpairs, num_celllines, num_drugs)
        
        self.train_hyperedges = \
            self._make_hyperedges(train_pairs, num_celllines)
        
        self.r = len(train_pairs[train_pairs[:, 2] == -1]) / len(train_pairs[train_pairs[:, 2] == 1])
        print(f'r: {self.r}')
        print(f'training pos edge: {len(train_pairs[train_pairs[:, 2] == 1])}')
        print(f'training neg edge: {len(train_pairs[train_pairs[:, 2] == -1])}')
    
    def _make_mask(self, indices, allpairs, num_celllines, num_drugs):
        """
        Create data mask
        
        Args:
            indices: data indices
            allpairs: cell line-drug pair data
            num_celllines: number of cell lines
            num_drugs: number of drugs
            
        Returns:
            torch.Tensor: flattened mask vector
        """
        index = allpairs[indices][:, 0:3]
        # adjust drug indices: subtract num_celllines to get the correct column 
        # indices (0 to num_drugs-1)
        drug_indices = index[:, 1] - num_celllines
        matrix = coo_matrix((np.ones(index.shape[0], dtype=bool), 
                           (index[:, 0], drug_indices)), 
                          shape=(num_celllines, num_drugs)).toarray()
        return torch.from_numpy(matrix).view(-1)
    
    def _make_pos_lable(self, allpairs, num_celllines, num_drugs):
        """
        Create positive sample labels
        
        Args:
            allpairs: cell line-drug pair data
            num_celllines: number of cell lines
            num_drugs: number of drugs
            
        Returns:
            torch.Tensor: positive sample label vector
        """
        pos_pairs = allpairs[allpairs[:, 2] == 1]
        # adjust drug indices to 0-based
        pos_pairs[:, 1] -= num_celllines
        label_pos = coo_matrix((np.ones(pos_pairs.shape[0]), 
                                (pos_pairs[:, 0], pos_pairs[:, 1])), 
                                shape=(num_celllines, num_drugs)
                                ).toarray()
        return torch.from_numpy(label_pos).type(torch.FloatTensor).view(-1)
    
    def _make_hyperedges(self, train_pairs, num_celllines):
        hyperedges = {
            'cellline': {},
            'drug': {}
        }
        cellline_pairs = np.array(train_pairs)
        cellline_pairs[:, 1] -= num_celllines
        cellline_pairs = cellline_pairs[np.argsort(cellline_pairs[:, 1])]
        cellline_positive = np.vstack([cellline_pairs[(cellline_pairs[:,2]==1) & (cellline_pairs[:,1]==drug)][:,:2] for drug in np.unique(cellline_pairs[:,1])])
        
        cellline_positive = torch.tensor(cellline_positive.T, dtype=torch.int64)

        cellline_negative = np.vstack([cellline_pairs[(cellline_pairs[:,2]==-1) & (cellline_pairs[:,1]==drug)][:,:2] for drug in np.unique(cellline_pairs[:,1])])
        
        cellline_negative = torch.tensor(cellline_negative.T, dtype=torch.int64)

        hyperedges['cellline']['positive'] = cellline_positive
        hyperedges['cellline']['negative'] = cellline_negative

        drug_pairs = cellline_pairs
        drug_pairs[:, [0, 1]] = drug_pairs[:, [1, 0]]
        drug_pairs = drug_pairs[np.argsort(drug_pairs[:, 1])]
        drug_positive = np.vstack([drug_pairs[(drug_pairs[:,2]==1) & (drug_pairs[:,1]==cellline)][:,:2] for cellline in np.unique(drug_pairs[:,1])])
        drug_positive = torch.tensor(drug_positive.T, dtype=torch.int64)

        drug_negative = np.vstack([drug_pairs[(drug_pairs[:,2]==-1) & (drug_pairs[:,1]==cellline)][:,:2] for cellline in np.unique(drug_pairs[:,1])])
        drug_negative = torch.tensor(drug_negative.T, dtype=torch.int64)

        hyperedges['drug']['positive'] = drug_positive
        hyperedges['drug']['negative'] = drug_negative

        return hyperedges
