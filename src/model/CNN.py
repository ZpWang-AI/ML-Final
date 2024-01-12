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
        hidden_channels,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        
        self.kernel_sizes = [8,4,3]
        self.cnn = nn.Sequential(
            CNNBlock(nn.Conv2d(1,hidden_channels,(self.kernel_sizes[0],data_dim)), dropout=dropout),
            CNNBlock(nn.Conv2d(hidden_channels,hidden_channels,(self.kernel_sizes[1],1)), dropout=dropout),
            CNNBlock(nn.Conv2d(hidden_channels, data_dim, (self.kernel_sizes[2],1)), norm=False)
        )
        self.train()
        
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def train(self, mode=True):
        super().train(mode=mode)
        if mode:
            dilation = 1
            for kernel_size, block in zip(self.kernel_sizes, self.cnn):
                conv = block.block[0]
                conv.dilation = (dilation, dilation)
                dilation *= kernel_size
                conv.stride = (1,1)
        else:
            for kernel_size, block in zip(self.kernel_sizes, self.cnn):
                conv = block.block[0]
                conv.stride = (kernel_size, kernel_size)
                conv.dilation = (1,1)
    
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