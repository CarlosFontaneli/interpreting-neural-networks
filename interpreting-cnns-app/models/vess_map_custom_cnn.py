import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = nn.ReLU()(out)
        return out


class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.resblock1 = ResidualBlock(64, 128, stride=2)
        self.resblock2 = ResidualBlock(128, 256, stride=2)
        self.resblock3 = ResidualBlock(256, 512, stride=2)
        self.resblock4 = ResidualBlock(512, 1024, stride=2)

        self.dropout = nn.Dropout(0.5)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )

        # Final Layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.resblock1(x1)
        x3 = self.resblock2(x2)
        x4 = self.resblock3(x3)
        x5 = self.resblock4(x4)

        x5 = self.dropout(x5)

        # Decoder
        x = self.upconv1(x5)  # Upsampling
        x = crop_and_add(x, x4)
        x = self.upconv2(x)
        x = crop_and_add(x, x3)
        x = self.upconv3(x)
        x = crop_and_add(x, x2)
        x = self.upconv4(x)
        x = crop_and_add(x, x1)

        # Final Layer
        x = self.final(x)

        return x


def crop_and_add(x1, x2):
    """Crop x2 to the size of x1 and add them."""
    B, C, H, W = x2.size()
    x1 = x1[:B, :C, :H, :W]
    return x1 + x2
