from typing import List

import torch
from torch_geometric.data import InMemoryDataset, Data

class GraphDataset(InMemoryDataset):
    """Graph dataset class compatible with PyTorch Geometric 2.6.1"""
    def __init__(self, root='.', dataset='davis', 
                 transform=None, pre_transform=None, 
                 graphs_dict=None, dttype=None
                 ):
        self.graphs_dict = graphs_dict
        self.dataset = dataset if dataset else 'davis'
        self.dttype = dttype
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        if graphs_dict is not None:
            self.process(graphs_dict)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [self.dataset + f'_data_{self.dttype}.pt'] \
                if self.dttype \
                else [f'{self.dataset}_data.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        """Process graph data"""
        if graphs_dict is None:
            return
            
        data_list = []
        for data_mol in graphs_dict:
            features, edge_index = data_mol[0], data_mol[1]
            GCNData = Data(
                x=torch.FloatTensor(features), 
                edge_index=torch.LongTensor(edge_index)
            )
            data_list.append(GCNData)
        self._data = data_list
        
        # 为了兼容性，创建一个属性来存储数据
        self._data_list = data_list

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]
        
    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)