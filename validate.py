import torch
import torchvision
from torchvision import transforms
import models

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = models.CAM_resnet18_CIFAR()

model.load_state_dict(torch.load('checkpoints/checkpoint_50.pt'))
model.to(device)
model.eval()

batch_size = 16

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

total = 0
total_correct = 0

for input, label in testloader:
    input = input.to(device)
    label = label.to(device)
    pred, _ = model(input)

    pred = torch.argmax(pred, dim=1)

    total += len(input)

    total_correct += torch.sum(pred == label)

print(f'Accuracy: {total_correct/total}')



