import torch
import torch.nn as nn
from unet import UNet
import torch.nn.functional as F


class W_Net(nn.Module):
    def __init__(self, input_dim, D_channels, E_channels):
        super(W_Net, self).__init__()

        self.input_dim = input_dim
        self.D_channels = D_channels
        self.E_channels = E_channels
        self.rgb_channels = 3
        self.DNet = UNet(self.input_dim, self.D_channels, 1)
        self.ENet = UNet(self.D_channels + self.rgb_channels, self.E_channels)

        self.D_outconv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.D_dropoutconv = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(64, self.D_channels, kernel_size=3, padding=1),
            # nn.ReLU()
        )
        self.E_dropoutconv = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(64, self.E_channels, kernel_size=3, padding=1)
        )

    def forward(self, rgb):
        x = self.DNet(rgb)
        pd_distmap = self.D_outconv(x)
        d_features = self.D_dropoutconv(x)

        #l2 norm embedding 편차를 줄여줌
        norm_d = F.normalize(d_features, p=2, dim=1)

        ENet_input = torch.cat((norm_d, rgb), dim=1)
        x = self.ENet(ENet_input)
        embedding = self.E_dropoutconv(x)
        norm_embedding = F.normalize(embedding)

        return (pd_distmap, norm_embedding)


