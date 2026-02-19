import torch.nn as nn

class LeNet5(nn.Module): 
    def __init__(self) -> None:
        super().__init__()

        # =================
        # Convolution Layer
        # =================

        self.feature_extractor = nn.Sequential(
            # ===========
            # First Layer 
            # ===========

            nn.Conv2d(
                in_channels=1, 
                out_channels=6, 
                kernel_size=5, 
                stride=1, 
                padding=0
            ), 
            nn.ReLU(), 
            nn.AvgPool2d(kernel_size=2, stride=2), 

            # ============
            # Second Layer 
            # ============

            nn.Conv2d(
                in_channels=6, 
                out_channels=16, 
                kernel_size=5, 
                stride=1, 
                padding=0
            ), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # =====================
        # Fully Connected Layer
        # =====================

        self.fully_connected = nn.Sequential(
            nn.Flatten(), 

            nn.Linear(
                in_features=16 * 5 * 5, 
                out_features=120
            ), 
            nn.ReLU(), 

            nn.Linear(
                in_features=120, 
                out_features=84
            ), 
            nn.ReLU(), 

            nn.Linear(
                in_features=84, 
                out_features=10 
            )
        )

    def forward(self, x): 
        x = self.feature_extractor(x)
        logits = self.fully_connected(x) 
        return logits 