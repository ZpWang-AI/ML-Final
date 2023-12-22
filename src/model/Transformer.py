import torch
import torch.nn as nn
from model.criterion import SMAPELoss, MSELoss


class Transformer(nn.Module):
    def __init__(
        self, 
        data_dim,
        channels,
        num_layers,
        nhead,
        dropout,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        
        self.encoder_projection = nn.Linear(data_dim, channels)
        self.decoder_projection = nn.Linear(data_dim, channels)
        
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
        
        self.linear = nn.Linear(channels, data_dim)
        # self.criterion = SMAPELoss(mean_dim=(0,1))
        self.criterion = MSELoss(mean_dim=(0,1))
    
    def get_pos_emb(self, emb, embed_layer):
        batch_size, sequence_len, channels = emb.shape
        pos_emb = torch.arange(0, sequence_len, device=emb.device)
        pos_emb = embed_layer(pos_emb)
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        return pos_emb
    
    def model_forward(self, src, tgt):
        """
        src: [batch size, src seq, data_dim]
        tgt: [batch size, tgt seq, data_dim]
        src_emb: [batch size, src seq, channels]
        tgt_emb: [batch size, tgt seq, channels]
        tgt_mask: [tgt seq, tgt seq]
        output: [batch size, tgt seq, channels/data_dim]
        """
        src_emb = self.encoder_projection(src)
        tgt_emb = self.decoder_projection(tgt)
        src_emb += self.get_pos_emb(src_emb, self.encoder_pos_embedding)
        tgt_emb += self.get_pos_emb(tgt_emb, self.decoder_pos_embedding)

        output = self.transformer(src=src_emb, tgt=tgt_emb,)
        output = self.linear(output)
        return output
    
    def forward(self, inputs):
        """
        inputs: [batch size, seq length, 8]
        src: [batch size, 96, data_dim]
        tgt: [batch size, 96+96/336, data_dim]
        output: [batch size, 96+96/336, data_dim]
        y: [batch size, 96/336, data_dim]
        pred: [batch size, 96/336, data_dim]
        """
        inputs = inputs[..., :self.data_dim]
        src = inputs[:, :96, ]
        tgt = inputs[:, 48:96, ]
        y = inputs[:, 96:, ]
        tgt = torch.concat([tgt, torch.zeros_like(y)], dim=1)
        output = self.model_forward(src, tgt)
        pred = output[:, 48:, ]
        loss = self.criterion(pred, y)
        return {'pred':pred, 'gt':y, 'loss':loss}
