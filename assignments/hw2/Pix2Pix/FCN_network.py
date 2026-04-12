import torch
import torch.nn as nn


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FullyConvNetwork(nn.Module):
    """
    U-Net style generator for 256x256 RGB image-to-image translation.
    Input : (B, 3, 256, 256)
    Output: (B, 3, 256, 256), range [-1, 1]
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = DownBlock(3, 64, normalize=False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1,
                      bias=False), nn.ReLU(inplace=True))

        # Decoder
        self.up1 = UpBlock(512, 512, dropout=True)
        self.up2 = UpBlock(1024, 512, dropout=True)
        self.up3 = UpBlock(1024, 512, dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        b = self.bottleneck(d7)

        # Decoder with skip connections
        u1 = self.up1(b)
        u1 = torch.cat([u1, d7], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)

        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)

        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        out = self.final(u7)
        return out


if __name__ == "__main__":
    model = FullyConvNetwork()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Output range:", y.min().item(), y.max().item())
