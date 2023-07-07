from torch import nn
import torch
class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Encoder,self).__init__()
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Down = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,3,2,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        out = self.Conv2(x)
        out_ = self.Down(out)
        return out,out_

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Decoder, self).__init__()
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
        )
        self.Up=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels*2,out_channels=out_channels,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x,out):
        x=self.Conv2(x)
        x=self.Up(x)
        out=torch.cat([x,out],dim=1)
        return out

class uNet(nn.Module):
    def __init__(self):
        super(uNet, self).__init__()
        self.d1=Encoder(3,64)
        self.d2=Encoder(64,128)
        self.d3=Encoder(128,256)
        self.d4=Encoder(256,512)
        self.u1=Decoder(512,512)
        self.u2=Decoder(1024,256)
        self.u3=Decoder(512,128)
        self.u4=Decoder(256,64)

        self.final=nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,1,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.final(out8)
        return out

"""
x=torch.rand(1,3,256,256)
net=uNet()
x=net(x)
print(x.shape)
"""