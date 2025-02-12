
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x): # torch.Size([b, 3, 256, 256])
        x1 = self.inc(x) # torch.Size([b, 64, 256, 256])
        x2 = self.down1(x1) # torch.Size([b, 128, 128, 128])
        x3 = self.down2(x2) # torch.Size([b, 256, 64, 64])
        x4 = self.down3(x3) # torch.Size([b, 512, 32, 32])
        x5 = self.down4(x4) # torch.Size([b, 1024, 16, 16])
        x = self.up1(x5, x4) # torch.Size([b, 512, 32, 32])
        x = self.up2(x, x3) # torch.Size([b, 256, 64, 64])
        x = self.up3(x, x2) # torch.Size([b, 128, 128, 128])
        x = self.up4(x, x1) # torch.Size([b, 64, 256, 256])
        logits = self.outc(x) # torch.Size([b, 21, 256, 256])
        return logits
# 上采样, up
def forward(self, x1, x2):
    x1 = self.up(x1) # ConvTranspose2d
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

self.outc
OutConv(
  (conv): Conv2d(64, 21, kernel_size=(1, 1), stride=(1, 1))
)