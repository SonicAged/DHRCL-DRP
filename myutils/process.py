from torch_geometric.data import Batch

def collate(data_list):
    """Batch processing function compatible with PyG 2.6.1"""
    if not data_list:
        return None
    batchA = Batch.from_data_list(data_list)
    return batchA