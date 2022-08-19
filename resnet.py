import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchsummary import summary
import random
import os
import torch
import torchvision

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
SEED = 2022
fix_seed(SEED)

config = {
    "batch_size": 500,
    "learning_rate": 0.008,
    "epochs": 100,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config['batch_size'], shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='./cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config['batch_size'], shuffle=False)


#import tensorflow as tf

#from tensorflow.keras import datasets   #, layers, models 
#SEED = 2021
#os.environ['PYTHONHASHSEED'] = str(SEED)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)




class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=False, outputs=1000):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,), resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)

        return input
#resnet18
#resnet18 = ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1000)
#resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet18, (3, 224, 224))

## resnet34
#resnet34 = ResNet(3, ResBlock, [3, 4, 6, 3], useBottleneck=False, outputs=1000)
#resnet34.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet34, (3, 224, 224))
#
## resnet50
#resnet50 = ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=1000)
#resnet50.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet50, (3, 224, 224))
#
## resnet101
#resnet101 = ResNet(3, ResBottleneckBlock, [3, 4, 23, 3], useBottleneck=True, outputs=1000)
#resnet101.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet101, (3, 224, 224))
#
## resnet152
#resnet152 = ResNet(3, ResBottleneckBlock, [3, 8, 36, 3], useBottleneck=True, outputs=1000)
#resnet152.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
#summary(resnet152, (3, 224, 224))
#











#random.seed(SEED)
#numpy.random.seed(SEED)
#tf.random.set_seed(SEED)

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

## Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0





def compile(model, train_loader, test_loader, input_shape, epochs, optimizer, loss):
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for epoch in range(epochs):
        ###############################################################
        num_train = 0
        num_correct_train = 0
        print("\repoch", epoch+1)
        for (xList, yList) in train_loader:
            xList, yList = torch.autograd.Variable(
                xList), torch.autograd.Variable(yList)
            optimizer.zero_grad()

            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
                device = torch.device(config['device'])
                model.to(device)

            outputs = model(xList)
            train_loss_func = loss(outputs, yList)
            train_loss_func.backward()
            optimizer.step()

            num_train += len(yList)  # i.e., add bath size

            # torch.max() return a list where list[0]: val list[1]: index
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_train += (predicts == yList).float().sum()
            
            
            print('\r %d ...' % num_train, end='')
        
        train_acc.append(num_correct_train / num_train)
        train_loss.append(train_loss_func.data)
        print("\r    - train_acc %.5f train_loss %.5f" %
                  (train_acc[-1], train_loss[-1]))

        ###############################################################
        num_test = 0
        num_correct_test = 0
        for (xList, yList) in test_loader:
            if torch.cuda.is_available():
                xList = xList.type(torch.cuda.FloatTensor)
                yList = yList.type(torch.cuda.LongTensor)
                device = torch.device(config['device'])
                model.to(device)
            
            outputs = model(xList)
            test_loss_func = loss(outputs, yList)

            num_test += len(yList)
            predicts = torch.max(outputs.data, 1)[1]
            num_correct_test += (predicts == yList).float().sum()

        test_acc.append(num_correct_test / num_test)
        test_loss.append(test_loss_func.data)
        print("\r    - test_acc  %.5f test_loss  %.5f" %
                (test_acc[-1], test_loss[-1]))
    return train_loss, train_acc, test_loss, test_acc


resnet18 = ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=1000)
model = resnet18
loss = torch.nn.CrossEntropyLoss()

result = compile(model, trainloader, testloader, (-1, 3, 32, 32), config["epochs"], optimizer=torch.optim.Adam(
    model.parameters(), lr=config["learning_rate"]), loss=loss)
model_path = '/workspace/'+'resnet_base'+'/seed_model_'+str(SEED)+'.pth'
torch.save(model, model_path)
