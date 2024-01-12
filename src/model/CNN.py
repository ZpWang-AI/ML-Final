import torch
import torch.nn as nn
from model.criterion import MSELoss


class CNNBlock(nn.Module):
    def __init__(self, conv, norm=True, dropout=0):
        super().__init__()
        
        self.norm = nn.BatchNorm2d(conv.out_channels) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        
        self.block = nn.Sequential(
            conv,
            self.norm,
            nn.ReLU(),
            self.dropout,
        )    
        
    def forward(self, x):
        return self.block(x)


class CNN(nn.Module):
    def __init__(
        self,
        data_dim,
        num_layers,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        
        blocks = [
            CNNBlock(nn.Conv2d(1,1,(8,data_dim)), dropout=dropout),
            CNNBlock(nn.Conv2d(1,1,(4,1),dilation=8), dropout=dropout),
        ]
        for _ in range(num_layers):
            blocks.append(CNNBlock(nn.Conv2d(1,1,(3,1),padding=(1,0),), dropout=dropout))
        blocks.append(CNNBlock(nn.Conv2d(1, data_dim, (3,1), dilation=32), norm=False))
        self.cnn = nn.Sequential(*blocks)
        
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        y:      [batch size, 96/336,     data_dim]
            train
        x:      [batch size, 1,        96+96/336, data_dim]
        h:      [batch size, data_dim, 96/336+1,  1]
        pred:   [batch size, 96/336,   data_dim]
            eval
        x:      [batch size, 1,           96,       data_dim]
        h:      [batch size, data_dim,    1,        1]
        pred:   [batch size, 1 -> 96/336, data_dim]
        """ 
        inputs = inputs[..., :self.data_dim]
        y = inputs[:, 96:, ]
        
        if self.training:
            x = inputs.unsqueeze(1)
            h = self.cnn(x)
            pred = h.transpose(1,2).squeeze(-1)[:, :-1, ]
        else:
            x = inputs[:, :96, ].unsqueeze(1)
            pred = torch.zeros((inputs.shape[0], 0, self.data_dim))
            for _ in range(y.shape[1]):
                h = self.cnn(x)
                pred = torch.concat((pred, h.transpose(1,2).squeeze(-1)), dim=1)
                x = torch.concat((x[:, :, 1:, ], h.transpose(1,3)), dim=2)
        
        loss = self.criterion(pred, y)
        return {'pred':pred, 'gt':y, 'loss':loss}