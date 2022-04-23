import os
import torch
import argparse
import torchvision


import model

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
args = parser.parse_args()

device=args.device

m=model.MyModel()
m.train(True)
m=m.to(device)

batch_size=64
epochs=20
lr=0.01

transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0),(1)),
])

dataset=torchvision.datasets.FashionMNIST("./dataset",train=True,transform=transform,download=True)
test_data_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False)

m.load_state_dict(torch.load('weight.pth',map_location=device))
m.eval()

def calc_acc(preds,labels):
  _,pred_max=torch.max(preds,1)
  acc=torch.sum(pred_max==labels.data,dtype=torch.float64)/len(preds)
  return acc

def calc_loss(pred,labels):
  # _,pred_max=torch.max(pred,1)
  loss_function=torch.nn.CrossEntropyLoss()
  loss=loss_function(pred,labels)
  return loss

loss=0
acc=0
for images,labels in test_data_loader:
  images=images.to(device)
  labels=labels.to(device)

  preds=m(images)
  # print(pred,labels)
  acc += calc_acc(preds, labels)
  loss+=calc_loss(preds,labels)
  
loss=loss/len(test_data_loader)
acc = acc / len(test_data_loader)
print(f"accuracy: {acc}")
print(f"loss: {loss}")
