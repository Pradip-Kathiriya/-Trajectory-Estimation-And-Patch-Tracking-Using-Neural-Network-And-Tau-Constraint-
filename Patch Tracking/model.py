
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        
        self.layer1 = nn.Sequential(nn.Conv2d(6,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh())                     
        self.layer2 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh())
        self.layer4 = nn.Sequential(nn.Conv2d(32,32,3,padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(32,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh())
        self.layer6 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh())
        self.layer8 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.Tanh(),
                                    nn.MaxPool2d(2))
        
        self.fc1 = nn.Linear(4096,1024)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(1024,64)
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(64, 3)

    def forward(self,x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.fc1(out.reshape((len(out),) + (torch.numel(out[0]),)))
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
        