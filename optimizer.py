"""Optimizer module for training deep learning models on drug-cell line interaction data."""

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from myutils.optimize import metrics_graph


class Optimizer(object):
    def __init__(
            self, model, drug_set, cellline_set, sampler,
            lr=1e-3, epochs=2000, alpha=0.2, beta=0.3,
            save_model=False, save_path=None, device="cuda:0",
            save_tensor=False
    ):
        """Initialize the optimizer for training the model.
        
        Args:
            model: Model instance to train
            drug_set: Drug dataset
            cellline_set: Cell line dataset
            sampler: Data sampler with train_mask, test_mask, train_edge, label_pos
            lr: Learning rate
            epochs: Number of training epochs
            alpha: Weight parameter for cellline contrastive loss
            beta: Weight parameter for drug contrastive loss
            save_model: Whether to save the best model
            save_path: Model save path
            save_tensot: Whether to save the tensors for data analysis
            device: Computation device
        """
        self.model = model
        self.drug_set = drug_set
        self.cellline_set = cellline_set
        self.sampler = sampler
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.save_model = save_model
        self.save_path = save_path
        self.save_tensor = save_tensor
        self.device = torch.device(device)

        # initialize the optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=0.00001
        )
        
        # initialize the scaler for mixed precision training
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        
        self.label_pos = sampler.label_pos.to(device)
        self.train_edge = torch.tensor(sampler.train_edge, device=device)
        
        # set the evaluation interval (evaluate every N epochs)
        self.eval_interval = 2
        
    def to(self, device):
        """move to the specified device"""
        self.device = torch.device(device)
        self.model = self.model.to(device)
        self.label_pos = self.label_pos.to(device)
        return self
        
    def __call__(self, progress_desc="Training Progress"):
        """Execute the training process.
        
        Args:
            progress_desc: Description for the progress bar
            
        Returns:
            best_epoch: Best training epoch
            test_true: Test set true labels
            test_pred: Test set predictions
            final_metrics: Best model evaluation metrics
            loss_list: Loss values during training
        """
        best_auc = 0
        best_epoch = 0
        final_metrics = {}
        test_true = None
        test_pred = None
        # Initialize dictionary to store all loss components
        loss_list = {
            'total_loss': [],
            'pos_loss': [],
            'contrastive_loss': [],
            'drug_sensitive': [],
            # 'drug_resistant': [],
            'cellline_sensitive': [],
            # 'cellline_resistant': []
        }
        
        # Initialize dictionary to store AUC values for each epoch
        auc_list = {
            'epoch': [],
            'auc': [],
            'aupr': [],
            'f1': [],
            'acc': [],
            'precision': [],
            'recall': [],
            'mcc': []
        }
        
        # initialize the current test result variable, 
        # avoid possible undefined errors
        curr_test_true = None
        curr_test_pred = None
        
        # training loop, use tqdm to display progress
        for epoch in tqdm(range(self.epochs), desc=progress_desc):
            for batch, (drug, cell) \
                in enumerate(zip(self.drug_set, self.cellline_set)):
                # move data to device
                drug.batch = drug.batch.to(self.device)
                mutation_data = cell[0].to(self.device)
                gexpr_data = cell[1].to(self.device)
                methylation_data = cell[2].to(self.device)

                # training step - use memory optimization strategy 
                # and mixed precision
                self.model.train()
                
                # clear the gradient, ensure a new round of training
                self.optimizer.zero_grad()
                
                # Forward pass with loss calculation - use mixed precision
                with autocast(enabled=self.use_amp):
                    _, _, _, losses = self.model(
                        drug=drug,
                        gexpr_data=gexpr_data,
                        methylation_data=methylation_data,
                        mutation_data=mutation_data,
                        edge=self.train_edge, 
                        hyperedges=self.sampler.train_hyperedges,
                        labels=self.label_pos,
                        train_mask=self.sampler.train_mask,
                        epoch=epoch
                    )
                    
                    
                    # Combine losses
                    loss = losses['total_loss']
                
                # backward propagation and optimization using mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if epoch % (self.eval_interval * 25) == 0:
                    tqdm.write(
                        f'Epoch {epoch}: \n'
                        f'pos loss={losses["pos_loss"]:.4f}\n'
                        f'contrastive loss={losses["contrastive_loss"]:.4f}\n'
                        f'\tdrug sensitive={losses["drug_sensitive"]:.4f}, '
                        # f'drug resistant={losses["drug_resistant"]:.4f}\n'
                        f'\tcellline sensitive={losses["cellline_sensitive"]:.4f}\n'
                        # f'cellline resistant={losses["cellline_resistant"]:.4f}\n'
                        f'total loss={losses["total_loss"]:.4f}'
                    )
                
                # Reduce evaluation frequency to specific intervals
                if epoch % self.eval_interval == 0 or epoch == self.epochs - 1:
                    # Evaluation step
                    with torch.no_grad():  # Ensure no computation graph is created
                        self.model.eval()
                        
                        # Ensure batch indices are correct
                        if not hasattr(drug, 'batch') or drug.batch is None:
                            # Create batch indices using vectorized operations
                            num_nodes = drug.x.size(0)
                            num_graphs = 1
                            if hasattr(drug, 'num_graphs'):
                                num_graphs = drug.num_graphs
                            elif hasattr(drug, 'ptr'):
                                num_graphs = len(drug.ptr) - 1
                                
                                # Create batch using ptr information
                                ptr = drug.ptr.to(self.device)
                                lengths = ptr[1:] - ptr[:-1]
                                if ptr.shape[0] > 0 and num_graphs > 0:
                                    last_length = num_nodes - ptr[-1]
                                    if last_length > 0:
                                        lengths = torch.cat([
                                            lengths, 
                                            torch.tensor([last_length], device=self.device)
                                        ])
                                
                                # Create batch indices using interleave
                                drug.batch = torch.repeat_interleave(
                                    torch.arange(num_graphs, device=self.device),
                                    lengths
                                )
                            else:
                                # Default to single graph
                                drug.batch = torch.zeros(
                                    num_nodes,
                                    dtype=torch.long,
                                    device=self.device
                                )
                        
                        # Forward pass for evaluation 
                        # (no need for loss calculation)
                        pos_adj, res_cellline, res_drug, _ = self.model(drug=drug,
                                                 gexpr_data=gexpr_data,
                                                 methylation_data=methylation_data,
                                                 mutation_data=mutation_data,
                                                 edge=self.train_edge,
                                                 hyperedges=self.sampler.train_hyperedges,
                                                 epoch=epoch
                                                 )

                        # Test set evaluation - memory optimized
                        with torch.no_grad(): 
                            # apply sigmoid activation function, 
                            # because we removed it in the model
                            ytest_p = torch.sigmoid(pos_adj[self.sampler.test_mask].detach())
                            ytest_t = self.label_pos[self.sampler.test_mask].detach()
                            
                            # Save CPU versions for final results
                            ytest_t_cpu = ytest_t.cpu()
                            
                            # Save best tensors for potential saving later
                            pos_adj_best = pos_adj.clone().detach()
                            res_cellline_best = res_cellline.clone().detach() if res_cellline is not None else None
                            res_drug_best = res_drug.clone().detach() if res_drug is not None else None
                            
                            # Free unneeded tensors to save memory
                            del pos_adj
                            
                            # Clear CUDA cache to free fragmented memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Check for NaN values before metric calculation
                            if torch.isnan(ytest_p).any():
                                print(f"Warning: Found {torch.isnan(ytest_p).sum().item()} NaN values in predictions")
                                ytest_p = torch.nan_to_num(ytest_p, nan=0.5)
                                
                            # Calculate evaluation metrics
                            (test_auc, test_aupr,
                             test_f1, test_acc,
                             test_precision, test_recall,
                             test_mcc) = metrics_graph(ytest_t, ytest_p)
                            
                            # Store AUC and other metrics for this epoch
                            auc_list['epoch'].append(epoch)
                            auc_list['auc'].append(test_auc)
                            auc_list['aupr'].append(test_aupr)
                            auc_list['f1'].append(test_f1)
                            auc_list['acc'].append(test_acc)
                            auc_list['precision'].append(test_precision)
                            auc_list['recall'].append(test_recall)
                            auc_list['mcc'].append(test_mcc)
                                
                            # Save current batch test results for later use
                            curr_test_true = ytest_t_cpu.numpy()
                            curr_test_pred = ytest_p.cpu().numpy()
                            
                            # Further clean GPU memory
                            del ytest_p, ytest_t

                    if epoch % (self.eval_interval * 25) == 0:
                        tqdm.write(
                            f'Test - AUC: {test_auc:.4f}, '
                            f'AUPR: {test_aupr:.4f}, '
                            f'F1: {test_f1:.4f}, '
                            f'ACC: {test_acc:.4f}, '
                            f'MCC: {test_mcc:.4f}\n'
                        )

                # save the best model
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_epoch = epoch
                    
                    # save the best performance metrics
                    final_metrics = {
                        'auc': test_auc,
                        'aupr': test_aupr, 
                        'f1': test_f1,
                        'acc': test_acc,
                        'precision': test_precision,
                        'recall': test_recall,
                        'mcc': test_mcc
                    }
                    
                    # use the test results already saved on CPU
                    test_true = curr_test_true
                    test_pred = curr_test_pred
                    
                    # save the model
                    if self.save_model and self.save_path:
                        torch.save(self.model.state_dict(), self.save_path + '.model')
                
                    # save tensors if they exist and save_tensor is enabled
                    if self.save_tensor:
                        if 'pos_adj_best' in locals():
                            torch.save(pos_adj_best, self.save_path + f'_pos_adj.pt')
                            del pos_adj_best
                        if 'res_drug_best' in locals():
                            torch.save(res_drug_best, self.save_path + f'_res_drug.pt')
                            del res_drug_best
                        if 'res_cellline_best' in locals():
                            torch.save(res_cellline_best, self.save_path + f'_res_cellline.pt')
                            del res_cellline_best
                            
                # periodically clear CUDA cache to prevent memory fragmentation
                if torch.cuda.is_available() and epoch % 20 == 0:
                    torch.cuda.empty_cache()
            # Store all loss components
            for key, value in losses.items():
                if key in loss_list:
                    loss_list[key].append(value.item())
            
        if 'test_true' not in locals() or test_true is None:
            test_true = np.array([])
            test_pred = np.array([])
            if not final_metrics:
                final_metrics = {
                    'auc': 0.0, 'aupr': 0.0, 'f1': 0.0, 'acc': 0.0,
                    'precision': 0.0, 'recall': 0.0, 'mcc': 0.0
                }
            
        return best_epoch, test_true, test_pred, final_metrics, loss_list, auc_list
