import argparse
import wandb
import torch
import torchvision

from model import *
from test import *


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()

wandb.init(project="fashionMnist")

device=torch.device(args.device)
model=MyModel()
model=model.to(device)
model.train(True)

#hyper parameter
batch_size=64
epochs=20
lr=0.01

transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0),(1)),
])

dataset=torchvision.datasets.FashionMNIST("./dataset",train=True,transform=transform,download=True)
train_data_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

optimizer=torch.optim.Adam(model.parameters(),lr=lr)
loss_function=torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
  train_loss=0.0
  train_acc=0.0
  for images,labels in train_data_loader:
    images=images.to(device)
    labels=labels.to(device)

    optimizer.zero_grad()
    pred=model(images)
    loss=loss_function(pred,labels)
    loss.backward()
    optimizer.step()
    train_loss+=loss
    train_acc+=calc_acc(pred,labels)
  
  total_loss=train_loss/len(train_data_loader)
  total_acc=train_acc/len(train_data_loader)
  print(f"epochs: {epoch} , loss: {total_loss}, acc: {total_acc}")
  wandb.log({'epochs': epoch+1,
        'loss': total_loss,
        'accuracy':total_acc
        })

torch.save(model.state_dict(),"fashion.pth")


