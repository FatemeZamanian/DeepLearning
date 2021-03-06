import torch
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1=nn.Conv2d(3,16,(3,3),(1,1),(0,0))
    self.conv2=nn.Conv2d(16,32,(3,3),(1,1),(0,0))
    self.conv3=nn.Conv2d(32,64,(3,3),(1,1),(1,1))
    self.conv4=nn.Conv2d(64,128,(3,3),(1,1),(1,1))

    self.fc1=nn.Linear(128*3*3,256)
    self.fc2=nn.Linear(256,128)
    self.fc3=nn.Linear(128,32)
    self.fc4=nn.Linear(32,1)
  
  def forward(self,x):
    x=F.relu(self.conv1(x))
    x=F.max_pool2d(x,kernel_size=(2,2))

    x=F.relu(self.conv2(x))
    x=F.max_pool2d(x,kernel_size=(2,2))

    x=F.relu(self.conv3(x))
    x=F.max_pool2d(x,kernel_size=(2,2))

    x=F.relu(self.conv4(x))

    x=torch.flatten(x,start_dim=1)

    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=F.relu(self.fc3(x))
    x=self.fc4(x)
    return x