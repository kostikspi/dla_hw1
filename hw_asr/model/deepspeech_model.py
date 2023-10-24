from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential

from hw_asr.base import BaseModel
from hw_asr.model.GRU import GRU
from hw_asr.model.Conv2ReLU import Conv2dReLU


# class Conv2dReLU(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dropout=0.1, padding='same'):
#         super(Conv2dReLU, self).__init__()
#         # padding = int((kernel_size - 1) / 2)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels, out_channels, kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels, out_channels, kernel_size, stride=stride, padding=padding),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         res = x
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return res + x
#
#
# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.1):
#         super(GRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.batch_first = batch_first
#         self.dropout = nn.Dropout(dropout)
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=batch_first,
#             bidirectional=True
#         )
#         self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
#
#     def forward(self, x):
#         # x = self.batch_norm(self.dropout(x))
#         x = F.leaky_relu(self.dropout(x))
#         # x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=self.batch_first)
#         x, _ = self.gru(x)
#         # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)
#         return x


class DeepSpeechModel(BaseModel):
    def __init__(self,
                 n_feats,
                 n_class,
                 fc_hidden=512,
                 n_gru_layer=2,
                 n_cnn_layer=2,
                 gru_dim=512,
                 dropout=0.1,
                 in_channels=32,
                 out_channels=32,
                 *args,
                 **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        conv_layers = [Conv2dReLU(in_channels=32,
                                  out_channels=32,
                                  kernel_size=3,
                                  stride=1,
                                  dropout=dropout, padding='same') if i != 0
                       else nn.Conv2d(in_channels=1,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=1, padding='same') for i in range(n_cnn_layer)]
        self.conv = Sequential(*conv_layers)
        gru_layers = []
        for i in range(n_gru_layer):
            if i == 0:
                gru_layers.append(nn.Linear(n_feats * in_channels, gru_dim))
                gru_layers.append(GRU(input_size=fc_hidden,
                                      hidden_size=gru_dim,
                                      dropout=dropout,
                                      batch_first=True))
            else:
                gru_layers.append(GRU(input_size=gru_dim * 2,
                                      hidden_size=gru_dim,
                                      dropout=dropout))
        self.gru = Sequential(*gru_layers)
        self.fc = Sequential(
            nn.Linear(in_features=gru_dim * 2, out_features=fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=fc_hidden, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        # batch x features x time
        x = spectrogram.mT.unsqueeze(1)  # batch x channels x time x ft
        x = self.conv(x)  # batch x channels x ft x time
        x = self.gru(x.transpose(2, 3).flatten(1, 2).mT)  # batch x time x ft * channels
        x = self.fc(x)  # batch x time x classes
        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths
