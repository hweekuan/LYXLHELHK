# LHK - modified 202004190730

import torch
import numpy as np
from torchvision import datasets,transforms
import torch.nn.functional as F

# ===============================================================
class base_data:
    def __init__(self):
        return  # does nothing
    # ===========================================================
    # sample data_set with num_pts of points
    # ===========================================================
    def sample(self,data_set,num_pts):
  
        indices = [i for i in range(len(data_set))]
        np.random.shuffle(indices)
        indices = indices[:num_pts]
  
        data_set.data = data_set.data[indices]
        data_set.targets = data_set.targets[indices] 

        return data_set


# ===============================================================
# wrapper class to handle MNIST data set
# ===============================================================
class myMNIST(base_data):
    def __init__(self,imageL,train_pts=0,test_pts=0):

        trans = [transforms.Resize((imageL,imageL)),transforms.ToTensor()]

        self.train_set = datasets.MNIST('../MNISTdata', train=True, download=True,
                         transform=transforms.Compose(trans))
        self.test_set  = datasets.MNIST('../MNISTdata', train=False,
                         transform=transforms.Compose(trans))
        if train_pts > 0:
            if train_pts > len(self.train_set):
                print('available ',len(self.train_set))
                raise ValueError("ERROR: request more than MNIST set")
            self.train_set = self.sample(self.train_set,train_pts)

        if test_pts > 0:
            if test_pts > len(self.test_set):
                print('available ',len(self.test_set))
                raise ValueError("ERROR: request more than MNIST set")
            self.test_set = self.sample(self.test_set,test_pts)
#
# ===============================================================
# wrapper class to handle MNIST data set
# ===============================================================
class myCIFAR10(base_data):
    def __init__(self,imageL,train_pts=0,test_pts=0):

        trans = [transforms.Resize((imageL,imageL)),transforms.ToTensor()]

        self.train_set = datasets.CIFAR10('../CIFAR10data', train=True, download=True,
                         transform=transforms.Compose(trans))
        self.test_set  = datasets.CIFAR10('../CIFAR10data', train=False,
                         transform=transforms.Compose(trans))

        self.train_set.targets = np.asarray(self.train_set.targets)
        self.test_set.targets = np.asarray(self.test_set.targets)

        if train_pts > 0:
            if train_pts > len(self.train_set):
                print('available ',len(self.train_set))
                raise ValueError("ERROR: request more than MNIST set")
            self.train_set = self.sample(self.train_set,train_pts)

        if test_pts > 0:
            if test_pts > len(self.test_set):
                print('available ',len(self.test_set))
                raise ValueError("ERROR: request more than MNIST set")
            self.test_set = self.sample(self.test_set,test_pts)

# ===============================================================
# class to load generic data, MNIST, CIFAR10 or others
# data_set must follow the protocol of myMNIST class
#
class data_loader:

    def __init__(self,data_set):

        self.data_set = data_set

        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        self.train_loader = torch.utils.data.DataLoader(self.data_set.train_set,
                            batch_size=128, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.data_set.test_set,
                           batch_size=1024, shuffle=False, **kwargs)
# ===========================================================
