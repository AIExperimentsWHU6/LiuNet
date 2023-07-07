import os
import random
#偶尔的灰色图像是因为随机化不固定
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from dataset import train_dataloader
import torch.nn as nn
import torch
from GAN_uNet import GAN_uNet
from matplotlib import pyplot as plt
from LiuNet import LiuNet
#设种子，调网络（鉴别器），改权重
img_shape = (3, 256, 256)
device = torch.device('cuda')


random_seed=812
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


min_loss=1000
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = LiuNet()

    def forward(self, z):
        img = self.model(z)
        # print("after_model",img.shape)
        # img=img.view(img.size(0),*img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(65536, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 16),
            nn.Sigmoid(),
        )
        """
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        """

    def forward(self, img):
        img = img.squeeze(0)
        img_flat = img.view(img.size(0), -1)
        # img_flat(1,256*256)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

generator.cuda()
discriminator.cuda()
adversarial_loss.cuda()

# Configure data loader
dataloader = train_dataloader

learning_rate = 0.001
b1 = 0.98  # adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor
# ----------
#  Training
# ----------
n_epochs = 1
generator_loss_list = []
discriminator_loss_list = []

for epoch in range(n_epochs):
    for img, mask, name in dataloader:
        img = img.to(device)
        mask = mask.to(device)
        # masks are the real_images,imgs are noises that are going through process
        # Adversarial ground truths
        valid = Variable(torch.ones([1, 16]), requires_grad=False).cuda()
        fake = Variable(torch.zeros([1, 16]), requires_grad=False).cuda()

        # Configure input
        masks = Variable(img.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))))
        # z=torch.rand(1,3,256,256).to(device)

        # Generate a batch of images
        # print("img shape",img.shape)
        gen_imgs = generator(img).to(device)
        # print("the shape of generate mask",gen_imgs.shape)
        # print("the shape of valid",valid.shape)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        generator_loss_list.append(g_loss)

        if g_loss < min_loss:
            min_test_loss = g_loss
            torch.save(generator.state_dict(), 'attention_residual.pth')

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_pred = discriminator(mask)
        fake_pred = discriminator(gen_imgs.detach())

        real_loss = adversarial_loss(real_pred, valid)
        fake_loss = adversarial_loss(fake_pred, fake)
        d_loss = (real_loss + fake_loss)*1.44
        discriminator_loss_list.append(d_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] {} [D loss: %f] [G loss: %f]".format(name)
            % (epoch, n_epochs, d_loss.item(), g_loss.item())
        )

        img_tensor = gen_imgs.squeeze(0)
        # print(img_tensor.shape)
        threshold = 0.5
        binary_image = (img_tensor > threshold).float()
        current_dir = os.getcwd()
        save_path = os.path.join(current_dir, "Attention_Residual_GAN")
        save_image(binary_image, os.path.join(save_path, "{}.png").format(name))



plt.figure(1)
plt.title("generator's loss")
plt.subplot(211)
plt.plot(generator_loss_list)
plt.title("discriminator's loss")
plt.subplot(212)
plt.plot(discriminator_loss_list)
plt.show()
