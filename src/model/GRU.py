import torch
import torch.nn as nn


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
        self.criterion = nn.MSELoss(reduction='sum')
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, ...]
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
            loss = self.criterion(pred, y)/y.shape[0]
        else:
            x, y = inputs[:, :96, ], inputs[:, 96:, ]
            h, ht = self.gru(x)
            m = h[:, -1:, ]
            pred = self.classifier(m)
            while pred.shape[1] < y.shape[1]:
                h, ht = self.gru(pred[:, -1:, ], ht)
                pred = torch.concat([pred, self.classifier(h)], dim=1)
            loss = self.criterion(pred, y)/y.shape[0]
        return {'pred': pred, 'gt': y, 'loss': loss}


if __name__ == '__main__':
    from configs import GRUConfig
    sample_inputs = torch.rand((5, 96+336, 8))
    sample_net = GRU(**GRUConfig())
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
    sample_net.eval()
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
