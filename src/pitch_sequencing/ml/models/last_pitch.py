import torch
import torch.nn as nn

class LastPitchTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len=1000, dropout=0.1):
        super(LastPitchTransformerModel, self).__init__()
        # Hardcode padding_idx to 0 for padding tokens in sequence. Add paramater in future 
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)  # Assuming max sequence length < 1000
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        pos = torch.arange(0, src.size(1), dtype=torch.long, device=src.device).unsqueeze(0)
        src = src + self.pos_encoder(pos)
        output = self.transformer(src, src_key_padding_mask=src_mask)
        return self.fc(output[:, -1, :])  # Only use the last position for prediction
    