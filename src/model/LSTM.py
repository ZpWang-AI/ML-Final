import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.,) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=output_size,
        )
    
    def forward(self, x, label=None):
        """
        x: [batch size, seq length, input size]
        h: [batch size, seq length, input size]
        ht: [num layers, batch size, hidden size]
        ct: [num layers, batch size, hidden size]
        m: [batch size, hidden size]
        """
        h, (ht, ct) = self.lstm(x)
        m = h[:,-1,]
        # m = ht[-1,...]
        logits = self.classifier(m)
        print(m.shape, logits.shape)
        
        pass


if __name__ == '__main__':
    sample_x = torch.rand((5,100,7))
    sample_net = LSTM(7, 128, 3, 0.1)
    sample_net(sample_x)
