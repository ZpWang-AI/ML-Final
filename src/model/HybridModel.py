import torch
import torch.nn as nn
from model.criterion import MSELoss


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0) -> None:
        super().__init__()
        layers = [
            nn.Linear(in_features, out_features),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
        
        self.residual = in_features == out_features
        
    def forward(self, x):
        y = self.block(x)
        if self.residual:
            y += x
        return y


class Hybrid(nn.Module):
    def __init__(
        self,
        data_dim,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        
        self.cnn = None
        self.mlp = None
        self.encoder = None
        self.decoder = None
        self.pos_emb = None
        self.lstm = None
        self.fc = None
        
        # self.hybrid = nn.Sequentia
        
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def model_forward(self, x):
        """
        x:      [batch size, 96, data_dim]
        return: [batch size, 1,  data_dim]
        """
        pass
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        y:      [batch size, 96/336,     data_dim]
            train
        x:      [batch size, 96,          data_dim]
        pred:   [batch size, 0 -> 96/336, data_dim]
            eval
        x:      [batch size, 96,          data_dim]
        nxt:    [batch size, 1,           data_dim]
        pred:   [batch size, 0 -> 96/336, data_dim]
        """ 
        inputs = inputs[..., :self.data_dim]
        x = inputs[:, :96, ]
        y = inputs[:, 96:, ]
        pred = self.model_forward(x)
        
        loss = self.criterion(pred, y)
        return {'pred':pred, 'gt':y, 'loss':loss}