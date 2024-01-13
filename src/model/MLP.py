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
        
        if num_layers == 1:
            layers = [MLPBlock(data_dim*96, data_dim)]
        else:
            layers = [MLPBlock(data_dim*96, hidden_size, dropout)]
            layers.extend([MLPBlock(hidden_size, hidden_size, dropout)]*(num_layers-2))
            layers.append(MLPBlock(hidden_size, data_dim))
        self.mlp = nn.Sequential(*layers)
        
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def model_forward(self, x):
        """
        x:      [batch size, 96, data_dim]
        return: [batch size, 1,  data_dim]
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        output = self.mlp(x)
        return output.unsqueeze(1)
    
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
        y = inputs[:, 96:, ]
        _, y_len, _ = y.shape
        
        if self.training:
            pred = torch.concat(
                [self.model_forward(inputs[:, p:p+96, ])for p in range(y_len)],
                dim=1,
            )
        else:
            x = inputs[:, :96, ]
            pred = torch.zeros((y.shape[0], 0, self.data_dim), device=inputs.device)
            for _ in range(y_len):
                nxt = self.model_forward(x)
                pred = torch.concat((pred, nxt), dim=1)
                x = torch.concat((x[:, 1:, ], nxt), dim=1)
        
        loss = self.criterion(pred, y)
        return {'pred':pred, 'gt':y, 'loss':loss}