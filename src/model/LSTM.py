import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, data_dim, hidden_size, num_layers, dropout=0., ) -> None:
        self.lstm = nn.LSTM(
            input_size=data_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=data_dim,
        )
    
    def forward(self, x, label):
        """
        x: [batch size, seq length, data dim]
        h: [batch size, seq length, data dim]
        ht: [num layers, batch size, hidden dim]
        ct: [num layers, batch size, hidden dim]
        m: [batch size, data dim]
        """
        h, (ht, ct) = self.lstm(x)
        m = h[:,-1,]
        # m = ht[-1,...]
        logits = self.classifier(m)
        
        pass


if __name__ == '__main__':
    pass
