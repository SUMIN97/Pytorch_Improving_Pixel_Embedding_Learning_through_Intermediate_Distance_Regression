import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet


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
            nn.Dropout(0.5),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.D_dropoutconv = nn.Sequential(
            nn.Conv2d(64, self.D_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.E_dropoutconv = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(64, self.E_channels, kernel_size=3, padding=1),
            # nn.ReLU()
        )


    def forward(self, rgb):
        x = self.DNet(rgb)
        pd_distmap = self.D_outconv(x)
        d_features = self.D_dropoutconv(x)

        #l2 norm embedding 편차를 줄여줌
        norm_d = F.normalize(d_features, p=2, dim=1)
        # norm_d = self.l2_norm(d_features)

        ENet_input = torch.cat((norm_d, rgb), dim=1)
        x = self.ENet(ENet_input)
        embedding = self.E_dropoutconv(x)
        norm_embedding = F.normalize(embedding, p=2, dim=1)
        # norm_embedding = self.l2_norm(embedding)


        return (pd_distmap, norm_embedding)

"""
    def l2_norm(self, x):
        #x 는 (bs, c, h, w)
        copy_x = x.permute(0, 2, 3, 1)  # (bs, h, w, c)

        inv_norm = torch.rsqrt(torch.sum(copy_x ** 2, dim=3, keepdim=True) + 1e-12)
        norm_x = copy_x * inv_norm
        norm_x = norm_x.permute(0, 3, 1, 2).contiguous()  # (bs, c, h, w)

        return norm_x
"""
