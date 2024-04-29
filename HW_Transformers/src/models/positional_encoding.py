import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, max_len):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()
        
        pos_encoding = torch.zeros(max_len, emb_size) 
        positions = torch.arange(0, max_len).unsqueeze(1).float()

        div_value = torch.exp(-1 * torch.arange(0, emb_size, 2).float() / emb_size * math.log(10000.0))
        pos_encoding[:, 0::2] = torch.sin(positions * div_value)
        pos_encoding[:, 1::2] = torch.cos(positions * div_value)
        # pos_encoding = pos_encoding.unsqueeze(0)
        pos_encoding = pos_encoding.unsqueeze(1)

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor):
        """
        token_embedding - тензор матрицы эмбеддингов: (S, B, E)
        """
        return token_embedding + self.pos_encoding[:token_embedding.size(0)]
        # return token_embedding + self.pos_encoding[:, : token_embedding.size(1)]