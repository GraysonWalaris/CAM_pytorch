import torch
import torch.nn as nn
import PyTorch_CIFAR10.cifar10_models.vgg as cifar10_vgg_models
import PyTorch_CIFAR10.cifar10_models.resnet as cifar10_resnet_models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import ResNet18_Weights
    
class CAM_resnet18_CIFAR_head(nn.Module):
    def __init__(self):
        super(CAM_resnet18_CIFAR_head, self).__init__()
        self.global_average_pool = nn.AvgPool2d(kernel_size=8)

        self.fc = nn.Linear(128, 10, bias=False)

        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.global_average_pool(x).view(-1, 128)
        x = self.fc(x)
        x = self.softmax(x)

        return x
    
class CAM_resnet18_CIFAR(nn.Module):
    def __init__(self):
        super(CAM_resnet18_CIFAR, self).__init__()
        resnet18 = cifar10_resnet_models.resnet18(pretrained=True)

        return_nodes = {
            'layer2.1.bn2': 'output',
        }

        self.resnet18_backbone = create_feature_extractor(model=resnet18, return_nodes=return_nodes)

        self.resnet18_head = CAM_resnet18_CIFAR_head()
    
    def forward(self, x):
        features_maps = self.resnet18_backbone(x)['output']
        return self.resnet18_head(features_maps), features_maps
    
# class CAM_resnet18_head(nn.Module):
#     def __init__(self):
#         super(CAM_resnet18, self).__init__()
#         self.global_average_pool = nn.AvgPool2d(kernel_size=7)

#         self.fc = nn.Linear(512, 1000)

#     def forward(self, x):
#         pass

# class CAM_resnet18(nn.Module):
#     def __init__(self):
#         super(CAM_resnet18, self).__init__()
#         resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)

#         # freeze the weights of pretrained model
#         for _, param in resnet18.named_parameters():
#             param.requires_grad = False

#         return_nodes = {
#             'layer4.1.bn2': 'output',
#         }

#         self.resnet18_backbone = create_feature_extractor(model=resnet18, return_nodes=return_nodes)


#     def forward(self, x):
#         return self.resnet18_backbone(x)['output']
    
# class CAM_resnet18_head(nn.Module):
#     def __init__(self):
#         super(CAM_resnet18, self).__init__()
#         self.global_average_pool = nn.AvgPool2d(kernel_size=7)

#         self.fc = nn.Linear(512, 1000)

#     def forward(self, x):
#         pass

# class CAM_resnet18(nn.Module):
#     def __init__(self):
#         super(CAM_resnet18, self).__init__()
#         resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=ResNet18_Weights.DEFAULT)

#         # freeze the weights of pretrained model
#         for _, param in resnet18.named_parameters():
#             param.requires_grad = False

#         return_nodes = {
#             'layer4.1.bn2': 'output',
#         }

#         self.resnet18_backbone = create_feature_extractor(model=resnet18, return_nodes=return_nodes)

    

#     def forward(self, x):
#         return self.resnet18_backbone(x)['output']


