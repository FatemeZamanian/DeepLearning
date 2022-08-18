import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(3,160,(3,3),(1,1),(0,0))
    self.conv2=nn.Conv2d(160,64,(3,3),(1,1),(0,0))
    self.conv3=nn.Conv2d(64,32,(3,3),(1,1),(0,0))
    self.conv4=nn.Conv2d(1024,2048,(3,3),(1,1),(0,0))
    self.conv5=nn.Conv2d(2048,1024,(3,3),(1,1),(1,1))
    self.conv6=nn.Conv2d(1024,128,(3,3),(1,1),(1,1))

    self.fc1=nn.Linear(32*18*18,256)
    self.fc2=nn.Linear(256,64)
    self.fc3=nn.Linear(64,16)
    self.fc4=nn.Linear(16,8)
    self.fc5=nn.Linear(8,1)

    self.norm1=nn.BatchNorm2d(160)
    self.norm2=nn.BatchNorm2d(64)
    self.norm3=nn.BatchNorm2d(32)
  
  def forward(self,x):
    x=F.relu(self.conv1(x))
    x=self.norm1(x)
    x=F.avg_pool2d(x,kernel_size=(2,2))
  
    x=F.relu(self.conv2(x))
    x=self.norm2(x)
    x=F.avg_pool2d(x,kernel_size=(2,2))

    x=F.relu(self.conv3(x))
    x=self.norm3(x)
    x=F.avg_pool2d(x,kernel_size=(2,2))

    # x=F.relu(self.conv4(x))
    # x=F.avg_pool2d(x,kernel_size=(2,2))

    # x=F.relu(self.conv5(x))
    # x=F.max_pool2d(x,kernel_size=(2,2))

    # x=F.relu(self.conv6(x))
    # x=F.max_pool2d(x,kernel_size=(2,2))

    # x=F.relu(self.conv7(x))
    # x=F.max_pool2d(x,kernel_size=(2,2))

    x=torch.flatten(x,start_dim=1)

    x=F.relu(self.fc1(x))
    x=F.sigmoid(self.fc2(x))
    x=F.relu(self.fc3(x))
    x=F.relu(self.fc4(x))
    x=self.fc5(x)
    return x