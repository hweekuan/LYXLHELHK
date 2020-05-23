from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class netmlp(nn.Module):

    # ================================================
    def __init__(self,image_size,nchannels):
        super(netmlp, self).__init__()

        self.n_classes = 10
        self.fc1 = nn.Linear(image_size*image_size*nchannels, 128)
        self.fc2 = nn.Linear(128, self.n_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.g = None
        self.xsize = None

    # ================================================
    def forward(self, x):

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    # ================================================
    def predict2(self,x):
        logits = self.forward(x)
        p = torch.argmax(logits,dim=1)
        return p
    # ================================================
