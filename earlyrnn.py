import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
#from models.EarlyClassificationModel import EarlyClassificationModel
from torch.nn.modules.normalization import LayerNorm


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super(CNNFeatureExtractor, self).__init__()

        self.layernorm = LayerNorm(in_channels)
        
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),  # Add dropout
            
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),  
            
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),

            nn.AdaptiveAvgPool3d((None, 1, 1))
        )

        self.linear = nn.Linear(3*190*20, hidden_dim)
        
        
    def forward(self, x):
        # x_permuted = x.permute(0, 2, 3, 4, 1)  # Move channels to last dim
        # x_normed = self.layernorm(x_permuted)  # Apply LayerNorm
        # x_flattend = x.view(8, 80, -1)
        # features = self.linear(x_flattend)

        # x_normed = x_normed.permute(0, 4, 1, 2, 3)

        features = self.cnn(x)
        
        features = features.squeeze(-1).squeeze(-1)
        features = features.permute(0, 2, 1)
        
        return features

class EarlyRNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=64, nclasses=6, num_rnn_layers=2, dropout=0.2):
        super(EarlyRNN, self).__init__()

        # # input transformations
        # self.intransforms = nn.Sequential(
        #     nn.LayerNorm(input_dim), # normalization over D-dimension. T-dimension is untouched
        #     nn.Linear(input_dim, hidden_dims) # project to hidden_dims length
        # )

        self.intransforms = CNNFeatureExtractor(input_dim, hidden_dims)
        self.backbone = nn.LSTM(input_size=hidden_dims, hidden_size=hidden_dims, num_layers=num_rnn_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=False)

        # Heads
        self.classification_head = ClassificationHead(hidden_dims, nclasses)
        self.stopping_decision_head = DecisionHead(hidden_dims)

    def forward(self, x):
        x = self.intransforms(x)
        
        outputs, last_state_list = self.backbone(x)
        log_class_probabilities = self.classification_head(outputs)
        probabilitiy_stopping = self.stopping_decision_head(outputs)
        print(probabilitiy_stopping)

        return log_class_probabilities, probabilitiy_stopping

    @torch.no_grad()
    def predict(self, x):
        logprobabilities, deltas = self.forward(x)

        def sample_stop_decision(delta):
            dist = torch.stack([1 - delta, delta], dim=1)
            return torch.distributions.Categorical(dist).sample().bool()

        batchsize, sequencelength, nclasses = logprobabilities.shape

        stop = list()
        for t in range(sequencelength):
            # stop if sampled true and not stopped previously
            if t < sequencelength - 1:
                stop_now = sample_stop_decision(deltas[:, t])
                stop.append(stop_now)
            else:
                # make sure to stop last
                last_stop = torch.ones(tuple(stop_now.shape)).bool()
                if torch.cuda.is_available():
                    last_stop = last_stop.cuda()
                stop.append(last_stop)

        # stack over the time dimension (multiple stops possible)
        stopped = torch.stack(stop, dim=1).bool()

        # is only true if stopped for the first time
        first_stops = (stopped.cumsum(1) == 1) & stopped

        # time of stopping
        t_stop = first_stops.long().argmax(1)

        # all predictions
        predictions = logprobabilities.argmax(-1)
        

        # predictions at time of stopping
        predictions_at_t_stop = torch.masked_select(predictions, first_stops)

        return logprobabilities, deltas, predictions_at_t_stop, t_stop

class ClassificationHead(torch.nn.Module):

    def __init__(self, hidden_dims, nclasses):
        super(ClassificationHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, nclasses, bias=True),
            nn.LogSoftmax(dim=2))

    def forward(self, x):
        return self.projection(x)

class DecisionHead(torch.nn.Module):

    def __init__(self, hidden_dims):
        super(DecisionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, 1, bias=True),
            nn.Sigmoid()
        )

        # initialize bias to predict late in first epochs
        torch.nn.init.normal_(self.projection[0].bias, mean=-2, std=1e-1)


    def forward(self, x):
        output = self.projection(x).squeeze(2)
        return output
    

if __name__ == "__main__":
    model = EarlyRNN()
