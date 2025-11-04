import torch.nn as nn
import torchvision.models as models

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Model(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=512):
        super().__init__()
        #resnet = getattr(models, base_model)(weights=None)
        resnet = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = ProjectionHead(resnet.fc.in_features, out_dim)

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return z