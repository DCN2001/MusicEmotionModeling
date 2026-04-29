import torch.nn as nn

class FF_model(nn.Module):
    def __init__(self, input_dim=768):
        super(FF_model, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.FC(x)
