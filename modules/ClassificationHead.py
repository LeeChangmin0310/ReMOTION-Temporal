'''
#############################################
# ClassificationHead_v2: 
# He init -> LayerNorm -> 256 -> GELU -> Dropout -> 256 -> 2 -> Res connection
#############################################

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        # Normalize input embedding
        self.norm = nn.LayerNorm(input_dim)

        # MLP block with residual connection
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Linear projection to hidden dim
            nn.GELU(),                         # Non-linearity
            nn.Dropout(dropout),               # Regularization
            nn.Linear(hidden_dim, input_dim)   # Project back to input dim
        )

        # Final classification layer
        self.classifier = nn.Linear(input_dim, num_classes)

        self._init_weights()

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x) + x  # Residual connection
        return self.classifier(x)

    def _init_weights(self):
        # He (Kaiming) initialization for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

'''
#############################################
# ClassificationHead_v1
#############################################

import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Enhanced classifier head with residual MLP blocks and normalization.
    Designed for discriminative session-level embeddings (e.g., from attention pooling).
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout_rate=0.3):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        # He initialization for GELU + LayerNorm setup
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (B, D)
        return self.classifier(x) #(B, num_classes)


'''
class ClassificationHead(nn.Module):
    """
    Final classifier that fuses the aggregated session-level embedding
    and outputs emotion predictions.
    """
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        self.classifier.apply(ClassificationHead.init_weights)

    # Initialize weights using Kaiming initialization
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.classifier(x)
'''
'''
class ResidualMLPHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ResidualMLPHead, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )
        self.final = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x + self.block(x)  # Residual
        return self.final(x)
'''