import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PyTorch_CIFAR10.cifar10_models.resnet import resnet18

import models

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Working on {torch.cuda.get_device_name()}')

model = models.CAM_resnet18_CIFAR()
model.load_state_dict(torch.load('checkpoints/checkpoint_40.pt'))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

batch_size = 16

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# training loop
epochs = 200
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters())
model.train()
model = model.to(device)

for epoch in range(epochs):
    total_loss = 0
    for input, label in trainloader:
        
        input, label = input.to(device), label.to(device)
        
        pred, _ = model(input)
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
    
    print(f'Epoch: {epoch} -- Loss: {total_loss/len(trainloader)}')

    if epoch % 5 == 0 and epoch != 0:
        torch.save(model.state_dict(), f'checkpoints/checkpoint_{epoch+40}.pt')







