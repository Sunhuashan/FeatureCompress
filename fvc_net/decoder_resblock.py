from fvc_net.basic import *


class DecoderWithResblock(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.resb=ResBlock(in_planes, in_planes)
        self.deconv1=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv2=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv3=deconv(in_planes, out_planes, kernel_size=3, stride=2)

    def forward(self, x):
        x1=self.resb(x)
        x2=self.resb(x1)
        x3=self.resb(x2)
        x4=self.deconv1(x+x3)

        x5=self.resb(x4)
        x6=self.resb(x5)
        x7=self.resb(x6)
        x8=self.deconv2(x4+x7)

        x9=self.resb(x8)
        x10=self.resb(x9)
        x11=self.resb(x10)
        x12=self.deconv3(x11+x9)
        return x12




class DecoderWithResblock_flow(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.resb=ResBlock(in_planes, in_planes)
        self.deconv1=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv2=deconv(in_planes, in_planes, kernel_size=3, stride=2)
        self.deconv3=deconv(in_planes, out_planes, kernel_size=3, stride=2)

    def forward(self, x):
        x1=self.resb(x)
        x2=self.resb(x1)
        x3=self.resb(x2)
        x4=self.deconv1(x+x3)

        x5=self.resb(x4)
        x6=self.resb(x5)
        x7=self.resb(x6)
        x8=self.deconv2(x4+x7)

        x9=self.resb(x8)
        x10=self.resb(x9)
        x11=self.resb(x10)
        x12=self.deconv3(x8+x11)
        return x12