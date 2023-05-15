import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
import random
import models

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f'Working on {torch.cuda.get_device_name()}')

model = models.CAM_resnet18_CIFAR()

model.load_state_dict(torch.load('checkpoints/checkpoint_50.pt'))
model.to(device)
model.eval()

batch_size = 16

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

transform_show = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testset_show = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_show)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

idx = random.randint(0, 1000)
visualizing_image_show, _ = testset_show[idx]
visualizing_image, label = testset[idx]
visualizing_image = visualizing_image.view(-1, 3, 32, 32).to(device)

# get weights and get feature maps
weights = model.resnet18_head.fc.weight

class_weights = weights[label]

pred, feature_maps = model(visualizing_image)

heatmap = torch.zeros(8, 8).to(device)

# the heatmap is a summation of each feature map multiplied by the weights
# corresponding to that feature map and the class of the image
for idx, weight in enumerate(class_weights):
    feature_map = feature_maps[0][idx]
    feature_map = feature_map * weight
    heatmap += feature_map

extent = 0, 32, 0, 32
fig = plt.figure(frameon=False)

# get image as a numpy array (H, W, C) between [0, 1]
visualizing_image_show = visualizing_image_show.cpu().detach().numpy()
visualizing_image_show = visualizing_image_show / 2 + 0.5
visualizing_image_show = np.transpose(visualizing_image_show, (1, 2, 0))

im1 = plt.imshow(visualizing_image_show, cmap=plt.cm.gray, interpolation='nearest',
                 extent=extent)

im2 = plt.imshow(heatmap.cpu().detach(), cmap=plt.cm.viridis, alpha=.4, interpolation='bilinear',
                 extent=extent)

plt.show()

