import torch
import torch.nn as nn
from model.criterion import MSELoss


class LSTM(nn.Module):
    def __init__(
        self,
        data_dim,
        hidden_size,
        num_layers,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
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
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        x:      [batch size, seq length, data_dim]
        y:      [batch size, 96/336,     data_dim]
        h:      [batch size, seq length, data_dim]
        ht:     [num layers, batch size, hidden size]
        ct:     [num layers, batch size, hidden size]
        m:      [batch size, 96/336,     hidden size]
        pred:   [batch size, 96/336,     data_dim]
        """
        inputs = inputs[..., :self.data_dim]
        if self.training:
            x = inputs
            y = inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            m = h[:, 95:-1, ]
            pred = self.classifier(m)
            loss = self.criterion(pred, y)
        else:
            x, y = inputs[:, :96, ], inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            m = h[:, -1:, ]
            pred = self.classifier(m)
            while pred.shape[1] < y.shape[1]:
                h, (ht, ct) = self.lstm(pred[:, -1:, ], (ht, ct))
                pred = torch.concat([pred, self.classifier(h)], dim=1)
            loss = self.criterion(pred, y)
        return {'pred': pred, 'gt': y, 'loss': loss}


class LSTMResidual(nn.Module):
    def __init__(
        self,
        data_dim,
        hidden_size,
        num_layers,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
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
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        x:      [batch size, seq length, data_dim]
        y:      [batch size, 96/336,     data_dim]
        h:      [batch size, seq length, data_dim]
        ht:     [num layers, batch size, hidden size]
        ct:     [num layers, batch size, hidden size]
        m:      [batch size, 96/336,     hidden size]
        pred:   [batch size, 96/336,     data_dim]
        """
        inputs = inputs[..., :self.data_dim]
        if self.training:
            x = inputs
            y = inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            m = h[:, 95:-1, ]
            pred = self.classifier(m)
            pred += inputs[:, 95:-1, ]  # Residual Network
            loss = self.criterion(pred, y)
        else:
            x, y = inputs[:, :96, ], inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            m = h[:, -1:, ]
            pred = self.classifier(m)
            pred += x[:, -1:, ]  # Residual Network
            while pred.shape[1] < y.shape[1]:
                h, (ht, ct) = self.lstm(pred[:, -1:, ], (ht, ct))
                nxt_pred = self.classifier(h)
                nxt_pred += pred[:, -1:, ]  # Residual Network
                pred = torch.concat([pred, nxt_pred], dim=1)
            loss = self.criterion(pred, y)
        return {'pred': pred, 'gt': y, 'loss': loss}


class LSTMWithoutLinear(nn.Module):
    def __init__(
        self,
        data_dim,
        hidden_size,
        num_layers,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.lstm = nn.LSTM(
            input_size=data_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=data_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def forward(self, inputs:torch.Tensor):
        """
        inputs: [batch size, seq length, 8]
        x:      [batch size, seq length, data_dim]
        y:      [batch size, 96/336,     data_dim]
        h:      [batch size, seq length, data_dim]
        ht:     [num layers, batch size, hidden size]
        ct:     [num layers, batch size, hidden size]
        m:      [batch size, 96/336,     hidden size]
        pred:   [batch size, 96/336,     data_dim]
        """
        inputs = inputs[..., :self.data_dim]
        if self.training:
            x = inputs
            y = inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            h2, (ht2, ct2) = self.lstm2(h)
            pred = h2[:, 95:-1, ]
            loss = self.criterion(pred, y)
        else:
            x, y = inputs[:, :96, ], inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            h2, (ht2, ct2) = self.lstm2(h)
            pred = h2[:, -1:, ]
            while pred.shape[1] < y.shape[1]:
                h, (ht, ct) = self.lstm(pred[:, -1:, ], (ht, ct))
                h2, (ht2, ct2) = self.lstm2(h, (ht2, ct2))
                pred = torch.concat([pred, h2], dim=1)
            loss = self.criterion(pred, y)
        return {'pred': pred, 'gt': y, 'loss': loss}