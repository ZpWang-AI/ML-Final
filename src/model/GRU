import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, data_dim, hidden_size, num_layers, dropout=0., ) -> None:
        super().__init__()
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
    
    def forward(self, x, label=None):
        """
        x: [batch size, seq length, data dim]
        h: [batch size, seq length, data dim]
        ht: [num layers, batch size, hidden dim]
        m: [batch size, data dim]
        """
        h, ht = self.gru(x)
        m = h[:,-1,]
        # m = ht[-1,...]
        logits = self.classifier(m)
        print(m.shape, logits.shape)
        pass


if __name__ == '__main__':
    sample_x = torch.rand((5,100,7))
    sample_net = GRU(7, 128, 3, 0.1)
    sample_net(sample_x)
