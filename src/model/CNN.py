import torch
import torch.nn as nn
from model.criterion import MSELoss


class CNNBlock(nn.Module):
    def __init__(self, conv, norm=True, dropout=0):
        super().__init__()
        
        self.norm = nn.BatchNorm1d(conv.out_channels) if norm else nn.Identity()
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
        hidden_channels,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        
        self.kernel_sizes = [2,3,4,4]
        self.cnn = nn.Sequential(
            CNNBlock(nn.Conv1d(data_dim,        hidden_channels, self.kernel_sizes[0]), dropout=dropout),
            CNNBlock(nn.Conv1d(hidden_channels, hidden_channels, self.kernel_sizes[1]), dropout=dropout),
            CNNBlock(nn.Conv1d(hidden_channels, hidden_channels, self.kernel_sizes[2]), dropout=dropout),
            CNNBlock(nn.Conv1d(hidden_channels, data_dim,        self.kernel_sizes[3]), norm=False)
        )
        self.train()
        
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            dilation = 1
            for kernel_size, block in zip(self.kernel_sizes, self.cnn):
                conv = block.block[0]
                conv.dilation = dilation,
                dilation *= kernel_size
                conv.stride = 1,
        else:
            for kernel_size, block in zip(self.kernel_sizes, self.cnn):
                conv = block.block[0]
                conv.stride = kernel_size,
                conv.dilation = 1,
                
    def model_forward(self, x):
        """
        x:      [batch size, len,    data_dim]
        return: [batch size, len-95, data_dim]
        """
        x = x.transpose(1,2)
        output = self.cnn(x)
        return output.transpose(1,2)
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        y:      [batch size, 96/336,     data_dim]
            train
        x:      [batch size, 96+96/336, data_dim]
        h:      [batch size, 1+96/336,  data_dim]
        pred:   [batch size, 96/336,    data_dim]
            eval
        x:      [batch size, 96,          data_dim]
        h:      [batch size, 1,           data_dim]
        pred:   [batch size, 0 -> 96/336, data_dim]
        """ 
        inputs = inputs[..., :self.data_dim]
        y = inputs[:, 96:, ]
        
        if self.training:
            x = inputs
            h = self.model_forward(x)
            pred = h[:, :-1, ]
        else:
            x = inputs[:, :96, ]
            pred = torch.zeros((inputs.shape[0], 0, self.data_dim), device=inputs.device)
            for _ in range(y.shape[1]):
                h = self.model_forward(x)
                pred = torch.concat((pred, h), dim=1)
                x = torch.concat((x[:, 1:, ], h), dim=1)
        
        loss = self.criterion(pred, y)
        return {'pred':pred, 'gt':y, 'loss':loss}