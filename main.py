import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import sys
import os

from netmlp                   import netmlp
from myio                     import myio
from data_loader              import myMNIST 
from data_loader              import myCIFAR10
from data_loader              import data_loader

#================================================
def build_dataset(name,num_trian_pts,num_test_pts):

    if name=='MNIST':
        print('===> use MNIST')
        image_size = 28
        data_set = myMNIST(image_size,num_train_pts,num_test_pts)
        nchannels = 1
    else:
        print('===> use CIFAR10')
        image_size = 32
        data_set = myCIFAR10(image_size,num_train_pts,num_test_pts)
        nchannels = 3

    return data_set,nchannels,image_size
#================================================
def manual_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
#================================================

if __name__=='__main__':

    num_train_pts = 8192
    num_test_pts = 1024
    manual_seed(5012)
    data_set,nchannels,image_size = build_dataset('CIFAR10',
                                                  num_train_pts,
                                                  num_test_pts)
    data = data_loader(data_set)

    lr = 0.1
    nepoch = 100

    net = netmlp(image_size,nchannels)
    optimizer = optim.Adadelta(net.parameters(),lr)

    io = myio()

    sys.stdout.flush()

    lr_nsteps = 0
    for epoch in range(nepoch):

        if epoch%10==0:
            print('epoch ',epoch)

        net.train()
        for batch_idx, (x, label) in enumerate(data.train_loader):
            optimizer.zero_grad()
            output = net.forward(x)
            loss = net.criterion(output, label)
            loss.backward()
            optimizer.step()

        net.eval()
        # display the predictions
        for batch_idx, (x,label) in enumerate(data.test_loader):
            if batch_idx==0:
                with torch.no_grad(): 
                    pred = net.predict2(x)
                    io.stream_out(x,pred,label,epoch)              


