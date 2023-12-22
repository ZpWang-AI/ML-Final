import torch
import torch.nn as nn
from model.criterion import MSELoss


class GRU(nn.Module):
    def __init__(
        self,
        data_dim, 
        hidden_size,
        num_layers, 
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.gru = nn.GRU(
            input_size=data_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=data_dim,
        )
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        x: [batch size, seq length, input size]
        y: [batch size, 96/336, output size]
        h: [batch size, seq length, input size]
        ht: [num layers, batch size, hidden size]
        ct: [num layers, batch size, hidden size]
        m: [batch size, 96/336, hidden size]
        pred: [batch size, 96/336, output size]
        """
        inputs = inputs[..., :self.data_dim]
        if self.training:
            x = inputs
            y = inputs[:, 96:, ]
            h, ht = self.gru(x)
            m = h[:, 95:-1, ]
            pred = self.classifier(m)
            loss = self.criterion(pred, y)
        else:
            x, y = inputs[:, :96, ], inputs[:, 96:, ]
            h, ht = self.gru(x)
            m = h[:, -1:, ]
            pred = self.classifier(m)
            while pred.shape[1] < y.shape[1]:
                h, ht = self.gru(pred[:, -1:, ], ht)
                pred = torch.concat([pred, self.classifier(h)], dim=1)
            loss = self.criterion(pred, y)
        return {'pred': pred, 'gt': y, 'loss': loss}

