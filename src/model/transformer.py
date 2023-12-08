import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, data_dim, hidden_size, num_layers, nhead, dropout,) -> None:
        super().__init__()
        self.data_dim = data_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=data_dim,
            dim_feedforward=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=data_dim,
            dim_feedforward=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.criterion = nn.MSELoss(reduction='sum')
    
    def forward(self, inputs):
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
            h, (ht, ct) = self.lstm(x)
            m = h[:, 95:-1, ]
            pred = self.classifier(m)
            loss = self.criterion(pred, y)/y.shape[0]
        else:
            x, y = inputs[:, :96, ], inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            m = h[:, -1:, ]
            pred = self.classifier(m)
            while pred.shape[1] < y.shape[1]:
                h, (ht, ct) = self.lstm(pred[:, -1:, ], (ht, ct))
                pred = torch.concat([pred, self.classifier(h)], dim=1)
            loss = self.criterion(pred, y)/y.shape[0]
        return {'pred': pred, 'gt': y, 'loss': loss}


if __name__ == '__main__':
    from configs import TransformerConfig
    sample_inputs = torch.rand((5, 96+336, 8))
    sample_net = Transformer(**TransformerConfig())
    out = sample_net.encoder(sample_inputs)
    print(sample_net)
    print(out, out.shape)
    exit()
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
    sample_net.eval()
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
        