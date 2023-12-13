import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, mean_dim=-1) -> None:
        super().__init__()
        self.mean_dim = mean_dim
    
    def forward(self, pred:torch.Tensor, gt:torch.Tensor):
        loss = ((pred-gt)**2)
        return loss.mean(dim=self.mean_dim).sum()
    

class SMAPELoss(nn.Module):
    def __init__(self, mean_dim=-1) -> None:
        super().__init__()
        self.mean_dim = mean_dim
    
    def forward(self, pred:torch.Tensor, gt:torch.Tensor):
        loss = (pred-gt).abs() / (pred.abs()+gt.abs()+1e-8)
        return loss.mean(dim=self.mean_dim).sum()
    

if __name__ == '__main__':
    x = torch.rand((3,4,5))
    y = torch.rand((3,4,5))
    net = MSELoss((0,1))
    net = MSELoss((0,1,2))
    loss1 = net(x,y)
    loss2 = ((x-y)**2).mean(dim=(0,1)).sum()
    loss3 = ((x-y)**2).sum() / 12
    loss4 = ((x-y)**2).sum() / 60
    print(loss1, loss2, loss3, loss4)