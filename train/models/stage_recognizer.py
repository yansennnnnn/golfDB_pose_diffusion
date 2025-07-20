import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCNBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = lambda x: x[:, :, :-padding]
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = lambda x: x[:, :, :-padding]
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class StageRecognizer(nn.Module):
    def __init__(self, input_dim, num_stages, num_channels, kernel_size=2, dropout=0.2):
        super(StageRecognizer, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, input_dim, kernel_size=1) # Reduce to input_dim
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_stages)

    def forward(self, x):
        # x shape: (B, T, C, H, W) - T is number of frames
        batch_size, num_frames, C, H, W = x.shape
        
        # Extract features from each frame
        features = []
        for t in range(num_frames):
            frame_feature = self.feature_extractor(x[:, t, :, :, :])
            frame_feature = self.avgpool(frame_feature) # (B, input_dim, 1, 1)
            features.append(frame_feature.squeeze()) # (B, input_dim)

        # (B, T, input_dim) -> (B, input_dim, T) for TCN
        features = torch.stack(features, dim=2) 
        
        # TCN processing
        tcn_output = self.tcn(features) # (B, num_channels[-1], T)
        
        # Use the feature from the last time step for classification
        last_feature = tcn_output[:, :, -1]
        
        # Classifier
        output = self.classifier(last_feature) # (B, num_stages)
        return output 