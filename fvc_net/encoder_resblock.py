from fvc_net.basic import *


class EncoderWithResblock(nn.Sequential):
    def __init__(self, in_planes: int=128, out_planes: int=128):
        super().__init__()
        self.conv1=conv(in_planes, out_planes, kernel_size=3, stride=2)
        self.conv2=conv(out_planes, out_planes, kernel_size=3, stride=2)
        self.conv3 = conv(out_planes, out_planes, kernel_size=3, stride=2)
        self.resb=ResBlock(out_planes, out_planes)


    def forward(self, x):
        x1=self.conv1(x)
        x2=self.resb(x1)
        x3=self.resb(x2)
        x4=self.resb(x3)

        x5=self.conv2(x1+x4)
        x6=self.resb(x5)
        x7=self.resb(x6)
        x8=self.resb(x7)

        x9=self.conv3(x5+x8)
        x10=self.resb(x9)
        x11=self.resb(x10)
        x12=self.resb(x11)

        return x9+x12


class EncoderWithResblock_flow(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.conv1=conv(2, out_planes, kernel_size=3, stride=2)
        self.conv2=conv(out_planes, out_planes, kernel_size=3, stride=2)
        self.conv3 = conv(out_planes, out_planes, kernel_size=3, stride=2)
        self.conv4 = conv(out_planes, out_planes, kernel_size=3, stride=2)
        self.resb=ResBlock(out_planes, out_planes)


    def forward(self, x):
        x1=self.conv1(x)
        x2=self.resb(x1)
        x3=self.resb(x2)
        x4=self.resb(x3)

        x5=self.conv2(x1+x4)
        x6=self.resb(x5)
        x7=self.resb(x6)
        x8=self.resb(x7)

        x9=self.conv3(x5+x8)
        x10=self.resb(x9)
        x11=self.resb(x10)
        x12=self.resb(x11)
        x13 = self.conv3(x9 + x12)

        return x13
