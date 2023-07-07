from torch import nn
import torch
import torchvision

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,padding=3)

    def forward(self,x):
        avg_pool=torch.mean(x,dim=1,keepdim=True)
        max_pool,_=torch.max(x,dim=1,keepdim=True)
        pool=torch.cat([avg_pool,max_pool],dim=1)
        #pool=[1,2,256,256]
        attention=self.conv(pool)
        return x*attention

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super(AttentionGate, self).__init__()

        self.in_channels = in_channels
        self.gating_channels = gating_channels

        # 注意力权重生成器
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, gating_channels, 1),
            nn.BatchNorm2d(gating_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gating_channels, gating_channels, kernel_size=1),
            nn.BatchNorm2d(gating_channels),
            nn.Sigmoid()
        )

    def forward(self, x, gating_signal):
        # 生成注意力权重
        attention_weights = self.attention(gating_signal)

        # 特征加权
        attended_features = x * attention_weights

        # 返回加权后的特征
        return attended_features

class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Encoder,self).__init__()
        self.Conv2 = nn.Sequential(
            Attention(),
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

    def forward(self, x):
        out = self.Conv2(x)
        out_shape = out.size()
        trans = torchvision.transforms.Resize((out_shape[2] // 2, out_shape[3] // 2))
        out_resi = trans(out)
        out_ = self.Down(out)
        out_ = out_resi + out_
        return out, out_

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Decoder, self).__init__()
        self.Conv2 = nn.Sequential(
            Attention(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
        )
        self.Upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.a1 = AttentionGate(out_channels, out_channels)

    def forward(self, x, concat):
        y = self.Conv2(x)
        y = self.Upsample(y)
        weight = self.a1(concat, y)
        out = torch.concat((y, weight), dim=1)
        return out

class LiuNet(nn.Module):
    def __init__(self):
        super(LiuNet, self).__init__()
        self.d1=Encoder(3,64)
        self.d2=Encoder(64,128)
        self.d3=Encoder(128,256)

        #self.d4=Encoder(256,512)
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
        #print(out_1.shape)
        #print(out1.shape)
        out_2,out2=self.d2(out1)
        #print(out_2.shape)
        #print(out2.shape)
        out_3,out3=self.d3(out2)
        #print(out_3.shape)
        #print(out3.shape)
        out_4,out4=self.d4(out3)
        #print(out_4.shape)
        #print(out4.shape)
        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.final(out8)
        return out

