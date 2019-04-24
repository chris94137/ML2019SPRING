import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
import sys
import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_num = 200

def read_origin_data(input_dir):
    print('start reading data...')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    img_list = []
    for i in range(img_num):
        img = Image.open(os.path.join(input_dir, (3 - len(str(i))) * '0' + str(i) + '.png'))
        img = transform(img).numpy()
        img_list.append(img)
        print('img', i, 'done', end = '\r', flush = True)
    img_list = np.array(img_list)
    data = torch.FloatTensor(img_list)
    loader = DataLoader(data, batch_size = 1, shuffle = False, num_workers = 2)
    print('data read...')
    return loader

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon  *sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def plot_attack(x, num, output_dir):
#     x = np.moveaxis(x, 0, -1)
#     print(x.shape)
    x.save(os.path.join(output_dir, (3 - len(str(num))) * '0' + str(num) + '.png'))
#     new_x = np.clip((x + 1) * 255 / 2, 0, 255)
#     result = Image.fromarray(new_x.astype(np.uint8))
#     result.save('/content/ans_'+ (3 - len(str(num))) * '0' + str(num) + '.png')  
def inverse_transform(t):
    return t.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1))

if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    model.eval()
    img_num = 200
    epsilon = 0.04
    data_loader = read_origin_data(sys.argv[1])
    criterion = nn.CrossEntropyLoss()
    L_inf_sum = 0.
    cnt = 0
    acc_num = 0
    for data in data_loader:
        print('\nimg num   :', cnt)
        epoch_count = 0
        data.requires_grad = True
        zero_gradients(data)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        loss = criterion(output, init_pred[0])
        loss.backward()
        perturbed_data = data + epsilon * data.grad.sign_()
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        while init_pred.item() == final_pred.item() and epoch_count < 500:
            perturbed_data = perturbed_data + epsilon * data.grad.sign_()
            output = model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            epoch_count += 1
        perturbed_data = inverse_transform(perturbed_data)
        origin_data = inverse_transform(data)
        adv = perturbed_data.squeeze().detach().numpy()
        origin = origin_data.squeeze().detach().numpy()
        L_inf = np.max(adv - origin)
        L_inf = abs(L_inf) * 255
        L_inf_sum += L_inf
        new_img = torch.clamp(perturbed_data.squeeze(), 0, 1)
        img_trans = transforms.Compose([transforms.ToPILImage()])
        new_img = img_trans(new_img)
        plot_attack(new_img, cnt, sys.argv[2])
        print('previous  :', init_pred.item())
        print('after     :', final_pred.item())
#         print('real label:', labels[cnt].item())
        print('L infinity:', L_inf)
        if init_pred.item() != final_pred.item():
            acc_num += 1
        cnt += 1
        print('fail times:', cnt - acc_num)
        if cnt == 200:
            break
    print('====================================================================')
    print('acc rate  :', acc_num / cnt)
    print('L infinity:', L_inf_sum / cnt)
    
        

