"""Utility functions for model evaluation and metrics calculation."""
import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_geometric.data import Data
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Individual metric calculation functions have been integrated into the metrics_graph
# function to reduce memory usage and avoid redundant calculations

def _get_drugs(drug, device):
    drugs_x = unbatch(drug.x, drug.batch)
    drugs_edge_index = unbatch_edge_index(drug.edge_index, drug.batch)
    drugs = []
    for x, edge_index in zip(drugs_x, drugs_edge_index):
        drugs.append(Data(x=x, edge_index=edge_index).to(device))
    return drugs

def multi_scale_topk_pooling(drug, ratios=[0.75, 0.5, 0.25], device="cuda:0"):
    """
    使用Top-K Pooling生成多尺度子图掩码
    :param drug: PyG的Data对象，包含节点特征和边信息
    :param ratios: 各尺度保留节点的比例列表
    :return: subgraph_masks 各尺度子图的掩码列表
    """
    subgraph_masks = []
    drugs = _get_drugs(drug, device)

    for ratio in ratios:
        pool = TopKPooling(in_channels=drug.x.size(1), ratio=ratio).to(device)
        subgraph_mask = []
        for drug in drugs:
            x_pool, edge_index_pool, _, batch_pool, perm, _ = pool(drug.x, drug.edge_index)
            perms= torch.zeros(drug.x.shape[0], dtype=torch.long)
            perms[perm.tolist()] = 1
            subgraph_mask.append(perms)
        subgraph_masks.append(subgraph_mask)
    sub_batchs = [list(rt) for rt in zip(*subgraph_masks)]

    return sub_batchs

def metrics_graph(yt, yp):
    """Calculate various evaluation metrics for model performance assessment.
    
    Args:
        yt: True labels tensor
        yp: Predicted scores tensor
        
    Returns:
        Tuple containing various evaluation metrics:
        (auc, aupr, f1, accuracy, precision, recall, mcc)
    """
    # Reduce memory usage: move data to CPU immediately
    yt_cpu = yt.detach().cpu()
    yp_cpu = yp.detach().cpu()
    yt_np = yt_cpu.numpy()
    yp_np = yp_cpu.numpy()
    
    # Check for NaN values and replace them
    if np.isnan(yp_np).any():
        print(f"Warning: Found {np.sum(np.isnan(yp_np))} NaN predictions. Replacing with 0.5")
        yp_np = np.nan_to_num(yp_np, nan=0.5)
    
    # Calculate AUC and AUPR
    precision, recall, thresholds_np = precision_recall_curve(yt_np, yp_np)
    aupr = metrics.auc(recall, precision)
    auc = roc_auc_score(yt_np, yp_np)
    
    # Small constant to prevent division by zero
    epsilon = 1e-10
    
    # Calculate F1 scores for different thresholds
    f1_scores = 2 * precision * recall / (precision + recall + epsilon)
    max_f1_idx = np.argmax(f1_scores)
    f1 = f1_scores[max_f1_idx]
    
    # Ensure thresholds list has matching length
    if len(thresholds_np) < len(f1_scores):
        thresholds_np = np.append(thresholds_np, 0.0)
    optimal_threshold = thresholds_np[max_f1_idx]
    
    # Use numpy for other metrics to avoid creating large tensors on GPU
    y_pred_binary = (yp_np >= optimal_threshold).astype(float)
    
    # Calculate accuracy
    acc = np.mean(y_pred_binary == yt_np)
    
    # Calculate confusion matrix elements
    tp = np.sum(np.logical_and(y_pred_binary == 1, yt_np == 1))
    fp = np.sum(np.logical_and(y_pred_binary == 1, yt_np == 0))
    fn = np.sum(np.logical_and(y_pred_binary == 0, yt_np == 1))
    tn = np.sum(np.logical_and(y_pred_binary == 0, yt_np == 0))
    
    # More robust versions of precision and recall calculations
    # Precision = TP / (TP + FP)
    precision_score = tp / max(tp + fp, epsilon)
    # Recall = TP / (TP + FN)
    recall_score = tp / max(tp + fn, epsilon)
    
    # Calculate MCC with more stable method to avoid overflow
    # Convert to float to prevent integer overflow
    tp = float(tp)
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    
    # Use a more stable calculation approach to avoid overflow
    try:
        numerator = (tp * tn) - (fp * fn)
        # Use intermediate variables to reduce overflow risk
        term1 = tp + fp
        term2 = tp + fn
        term3 = tn + fp
        term4 = tn + fn
        
        # Check to avoid square root of zero or negative values
        if term1 <= 0 or term2 <= 0 or term3 <= 0 or term4 <= 0:
            mcc = 0.0  # MCC calculation is invalid when any term is zero
        else:
            denominator = np.sqrt(term1 * term2 * term3 * term4)
            if denominator > 0:
                mcc = numerator / denominator
            else:
                mcc = 0.0
    except (ValueError, OverflowError, ZeroDivisionError) as e:
        # Use fallback method or return 0 if error occurs
        print(f"Warning: MCC calculation error ({e}), returning 0")
        mcc = 0.0
    
    # Clean up memory
    del yt_cpu, yp_cpu
    
    return auc, aupr, f1, acc, precision_score, recall_score, mcc