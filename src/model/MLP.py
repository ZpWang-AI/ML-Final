import torch
import torch.nn as nn
from model.criterion import MSELoss


class MLP(nn.Module):
    def __init__(
        self,
        data_dim,
        hidden_size,
        num_layers,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.mlp = None
        self.seq_len_out = 96
        self._init_model(96, self.seq_len_out)
        
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def _get_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
    
    def _init_model(self, seq_len_in, seq_len_out):
        in_features = self.data_dim*seq_len_in
        out_features = self.data_dim*seq_len_out
        
        if self.num_layers == 1:
            layers = [
                nn.Linear(in_features, out_features),
                nn.ReLU(),
            ]
        else:
            layers = [self._get_block(in_features, self.hidden_size)]
            for _ in range(self.num_layers-1):
                layers.append(self._get_block(self.hidden_size, self.hidden_size))
            layers.append(nn.Linear(self.hidden_size, out_features))
            
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        x:      [batch size, 96,         data_dim]
        y:      [batch size, 96/336,     data_dim]
        input_x:[batch size, 96*data_dim]
        output: [batch size, 96/336*data_dim]
        pred:   [batch size, 96/336,     data_dim]
        """
        seq_len_in = 96
        seq_len_out = inputs.shape[1]-seq_len_in
        if seq_len_out != self.seq_len_out:
            self.seq_len_out = seq_len_out
            self._init_model(seq_len_in, seq_len_out)
            
        inputs = inputs[..., :self.data_dim]
        x = inputs[:, :96, ]
        y = inputs[:, 96:, ]
        
        input_x = x.reshape(-1, 96*self.data_dim)
        output = self.mlp(input_x)
        pred = output.reshape(-1, seq_len_out, self.data_dim)
        
        loss = self.criterion(pred, y)
        return {'pred':pred, 'gt':y, 'loss':loss}