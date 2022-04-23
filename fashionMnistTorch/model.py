import torch

class MyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=torch.nn.Linear(784,128)
    self.fc2=torch.nn.Linear(128,32)
    self.fc3=torch.nn.Linear(32,16)
    self.fc4=torch.nn.Linear(16,10)
    # self.fc5=torch.nn.Linear(64,32)
    # self.fc6=torch.nn.Linear(32,16)
    # self.fc7=torch.nn.Linear(16,10)
  def forward(self,x):
    x=x.reshape((x.shape[0],784))
    x=self.fc1(x)
    x=torch.relu(x)
    x=self.fc2(x)
    x=torch.relu(x)
    x=self.fc3(x)
    x=torch.relu(x)
    x=self.fc4(x)
    x=torch.relu(x)
    # x=self.fc5(x)
    # x=torch.relu(x)
    # x=self.fc6(x)
    # x=torch.relu(x)
    # x=self.fc7(x)
    # x=torch.relu(x)
    return x