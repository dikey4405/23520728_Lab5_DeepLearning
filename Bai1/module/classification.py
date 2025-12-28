import torch
import torch.nn as nn
from .transformer import TransformerEncoder, PositionnalEncoding, generate_padding_mask

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, head: int, n_layers: int, d_ff: int, dropout: float, vocab):
        super(TransformerModel, self).__init__()

        self.vocab = vocab
        
        self.embedding = nn.Embedding(vocab.vocab_size, d_model, padding_idx=vocab.pad_idx)
        self.PE = PositionnalEncoding(d_model, dropout)
        
        self.encoder = TransformerEncoder(d_model, head, n_layers, d_ff, dropout)

        self.ln_head = nn.Linear(d_model, vocab.num_labels)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        attention_mask = generate_padding_mask(input_ids, self.vocab.pad_idx).to(input_ids.device)

        input_embs = self.embedding(input_ids)
        features = self.PE(input_embs)
        features = self.encoder(features, attention_mask)

        features = features[:, 0, :]
        logits = self.dropout(self.ln_head(features))
        
        return logits, self.loss(logits, labels)
    
