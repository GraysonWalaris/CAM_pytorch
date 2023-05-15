# Description

This employs the techniques shown in the paper *Attention-Aware Generative Adversarial Networks (ATA-GANs)*. It shows how class activation maps can be used to show what portions of an image a convolutional neural network pays most attention to when making its prediction.

The official paper can be found here: https://arxiv.org/pdf/1802.09070.pdf.

# Method

For this, a resnet18 model pretrained on the CIFAR10 dataset was modified and finetuned on the CIFAR10 dataset. The modified model takes the feature maps from the output of the 2nd convolutional blocks, applies global average pooling to each feature map, and then uses a fully connected layer to output class probabilities. This modified architecture was retrained on the CIFAR10 datasets to learn the weights for the new fully connected layer.

The class activation maps can then be found and superimposed onto the image using the Soft CAM method proposed in the paper:

![alt text](https://github.com/gbbyrd/CAM_pytorch/blob/main/ref/CAM_diagram.png?raw=true)

# How to run

To finetune the modified resnet18 model, create a conda environment using:

```
conda env create -f environment.yml
```

Train the model using:

```
python train.py
```

Validate the performance (should be around 80% accuracy) of your training using:

```
python validate.py
```

# Visualizing the Results

To visualize the important parts of the image according to your model, use the following command:

```
python visualize.py
```

## Examples

![alt text](https://github.com/gbbyrd/CAM_pytorch/blob/main/ref/Bird.png?raw=true)

![alt text](https://github.com/gbbyrd/CAM_pytorch/blob/main/ref/Bird_2.png?raw=true)

![alt text](https://github.com/gbbyrd/CAM_pytorch/blob/main/ref/Car.png?raw=true)