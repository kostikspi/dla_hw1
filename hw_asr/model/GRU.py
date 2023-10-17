from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=True
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        # x = self.batch_norm(self.dropout(x))
        x = F.leaky_relu(self.dropout(x))
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=self.batch_first)
        x, _ = self.gru(x)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)
        return x
