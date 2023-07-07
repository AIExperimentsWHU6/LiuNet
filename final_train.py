import PIL.Image
from PIL.Image import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import dataset
import uNet_with_attention
import torch
import torch.optim as optim
import torch.nn as nn
import numpy
from torchvision import transforms
import torchvision.transforms.functional as F
from uNet_with_residual import uNet_residual
"""
torch.manual_seed(42)
train_loader = dataset.train_dataloader
test_loader = dataset.test_dataloader
epoch_num = 100
"""
#device_ids = [4,5,6,7]  # 指定可见的 GPU 设备 ID 列表
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(device_ids[0])  # 设置主 GPU 设备
model = uNet_residual().to(device=device)
#model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_loss = []
iou_list = []
iou_lists = []
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)

print(device)

def dice_score(output, target):
    smooth = 1e-5

    output_flat = output.view(-1)
    target_flat = target.view(-1)

    intersection = (output_flat * target_flat).sum()

    score = (2. * intersection + smooth) / (output_flat.sum() + target_flat.sum() + smooth)

    return score
def calculate_iou(A, B):
    intersection = (A * B).sum()
    union = A.sum() + B.sum() - intersection
    iou = intersection / union
    return iou
for epoch in range(epoch_num):
    running_loss = 0.0
    model.train()
    for i,data in enumerate(train_loader,0):
        input,label = data[0].to(device),data[1].to(device)
        output = model(input)
        model.zero_grad()

        loss = criterion(output,label).to(device)

        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        train_loss.append(loss.item())

    model.eval()
    for i,data in enumerate(test_loader,0):
        input,label = data[0].to(device),data[1].to(device)
        output = model(input)
        tensor_to_img = transforms.ToPILImage()

        x=input[0]
        y=label[0]
        pred=output[0]
        x_img = tensor_to_img(x)
        y_img = tensor_to_img(y)
        mask = (pred>0.5).float()
        output_mask = (output>0.5).float()
        iou = dice_score(output_mask,label).cpu()

        iou_list.append(iou.numpy())


        img_pil = transforms.ToPILImage()(mask)
        newImage = PIL.Image.new('RGB', (1200, 800))
        newImage.paste(x_img,(0,0))
        newImage.paste(y_img,(400,0))
        newImage.paste(img_pil,(800,0))
        newImage.save('./test/'+str(epoch)+'_'+str(i)+'.jpg')
    iou_lists.append(numpy.mean(iou_list))
    iou_list = []
    print('epoch:'+str(epoch)+' running_loss='+str(running_loss))
print('训练完成')
Path = './toy.pth'

plt.figure(1)
plt.subplot(121)
plt.title('loss')
plt.plot(train_loss)
plt.subplot(122)
plt.title('dice_score')
plt.plot(iou_lists)
plt.show()
plt.savefig('./fig.jpg')
torch.save(model.state_dict(),Path)