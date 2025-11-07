"""Model for deep learning on multi-omics data with drug-cell line interactions."""

import torch
import torch.nn as nn

class MyModel(nn.Module):
    """Main model integrating cell line and drug encoders with response prediction."""
    
    def __init__(
            self, hidden_channels, output_channels,
            cellline_encoder, drug_encoder, response_encoder,
            lambda_drug, lambda_cellline
    ):
        super(MyModel, self).__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.cellline_encoder = cellline_encoder
        self.drug_encoder = drug_encoder
        self.response_encoder = response_encoder
        self.lambda_drug = lambda_drug
        self.lambda_cellline = lambda_cellline
        # Initialize weight parameter on the correct device
        self.weight = nn.Parameter(torch.Tensor(hidden_channels, 
                                                hidden_channels)
                                    )
        
        self.fc = nn.Linear(self.output_channels, 10)
        self.fd = nn.Linear(self.output_channels, 10)
        self.batchc = nn.BatchNorm1d(self.output_channels)

    def to(self, device):
        super(MyModel, self).to(device)
        self.device = device
        self.cellline_encoder.to(device)
        self.drug_encoder.to(device)
        self.response_encoder.to(device)
        self.weight.to(device)
        self.fc.to(device)
        self.fd.to(device)
        self.batchc.to(device)        
        return self

    def forward(
            self, drug,
            gexpr_data, methylation_data, mutation_data, edge, 
            hyperedges, epoch, 
            labels=None, train_mask=None
    ):
        x_drug = self.drug_encoder(drug)
        x_cellline= self.cellline_encoder(
             gexpr_data, methylation_data, mutation_data
         )
        
        # Store the number of cell lines for later use
        index = x_cellline.shape[0]
        
        # Concatenate cell line and drug features
        feature = torch.cat((x_cellline, x_drug), 0)
        feature = self.batchc(feature)

        if not isinstance(edge, torch.Tensor):
            edge = torch.tensor(edge, device=self.device)
        elif edge.device != self.device:
            edge = edge.to(self.device)
            
        pos_edge = edge[edge[:, 2] == 1, 0:2].t().long()

        # Get response encoding
        pos_z, pos_drug, pos_cellline, losses = self.response_encoder(feature, pos_edge, hyperedges, index, epoch)

        # Split by cell lines and drugs
        cellpos = pos_z[:index, ]
        drugpos = pos_z[index:, ]

        # Get cell and drug features
        cellfea = self.fc(feature[:index, ])
        drugfea = self.fd(feature[index:, ])
        cellfea = torch.sigmoid(cellfea)
        drugfea = torch.sigmoid(drugfea)
        
        # Concatenate features
        cellpos = torch.cat((cellpos, cellfea, pos_cellline), 1)
        drugpos = torch.cat((drugpos, drugfea, pos_drug), 1)
        
        # Calculate adjacency matrix with NaN checking
        cellpos = torch.nan_to_num(cellpos, nan=0.0)
        drugpos = torch.nan_to_num(drugpos, nan=0.0)
        pos_adj = torch.matmul(cellpos, drugpos.t())
        
        # Calculate loss during forward pass if labels are provided
        if labels is not None and train_mask is not None:
            # Calculate supervised loss
            train_pos_adj = pos_adj.view(-1)[train_mask]
            train_label_pos = labels[train_mask]
            losses['pos_loss'] = self._loss_sup(train_pos_adj, train_label_pos)
            losses['contrastive_loss'] = self.lambda_drug * losses['contrastive_loss_drug'] \
                + self.lambda_cellline * losses['contrastive_loss_cellline']
            losses['total_loss'] = losses['pos_loss'] \
                + losses['contrastive_loss']
            
        return pos_adj.view(-1), cellpos, drugpos, losses if labels is not None else None
    
    def _loss_sup(self, pos_adj, label_pos):
        label_pos = label_pos.to(self.device)
        
        pos_loss = nn.BCEWithLogitsLoss()(pos_adj, label_pos)
        return pos_loss
    