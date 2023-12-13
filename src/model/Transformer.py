import torch
import torch.nn as nn
from model.criterion import SMAPELoss


class Transformer(nn.Module):
    def __init__(
        self, 
        encoder_dim,
        decoder_dim,
        channels,
        num_layers,
        nhead,
        dropout,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.encoder_projection = nn.Linear(encoder_dim, channels)
        self.decoder_projection = nn.Linear(decoder_dim, channels)
        
        self.encoder_pos_embedding = nn.Embedding(512, embedding_dim=channels)        
        self.decoder_pos_embedding = nn.Embedding(512, embedding_dim=channels)
        
        self.transformer = nn.Transformer(
            d_model=channels,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=channels*4,
            dropout=dropout,
            batch_first=True,
        )
        
        self.linear = nn.Linear(channels, encoder_dim-decoder_dim)
        self.criterion = SMAPELoss(mean_dim=(0,1))
    
    def get_pos_emb(self, emb, embed_layer):
        batch_size, sequence_len, channels = emb.shape
        pos_emb = torch.arange(0, sequence_len, device=emb.device)
        pos_emb = embed_layer(pos_emb)
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        return pos_emb
    
    def model_forward(self, src, tgt):
        src_emb = self.encoder_projection(src)
        tgt_emb = self.decoder_projection(tgt)
        final_src_emb = src_emb + self.get_pos_emb(src_emb, self.encoder_pos_embedding)
        final_tgt_emb = tgt_emb + self.get_pos_emb(tgt_emb, self.decoder_pos_embedding)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1], tgt.device)

        output = self.transformer(
            src=final_src_emb, 
            tgt=final_tgt_emb,
            tgt_mask=tgt_mask,
        )
        output = self.linear(output)
        return output
    
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
        src = inputs[:, :96, :self.encoder_dim]
        tgt = inputs[:, 96:, -self.decoder_dim:]
        y = inputs[:, 96:, :-self.decoder_dim]
        output = self.model_forward(src, tgt)
        loss = self.criterion(output, y)
        return {'pred':output, 'gt':y, 'loss':loss}
        if self.training:
            x = inputs
            y = inputs[:, 96:, ]
            h, (ht, ct) = self.lstm(x)
            m = h[:, 95:-1, ]
            pred = self.classifier(m)
            loss = self.criterion(pred, y)/y.shape[0]
        else:
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
    print(sample_net)
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
    sample_net.eval()
    sample_output = sample_net(sample_inputs)
    print(sample_output['pred'].shape, sample_output['gt'].shape, sample_output['loss'])
        