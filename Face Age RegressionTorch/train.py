import argparse
import wandb
import torch
import torchvision
import torchvision
from torchvision import transforms
from tqdm import tqdm



from model import *

def calc_acc(preds,labels):
  _,pred_max=torch.max(preds,1)
  acc=torch.sum(pred_max==labels.data,dtype=torch.float64)/len(preds)
  return acc


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--dataset", type=str,help='directory of dataset')
args = parser.parse_args()


device=torch.device(args.device)

device=torch.device(args.device if torch.cuda.is_available() else "cpu")
model=Model().to(device)

model.train(True)

#hyper parameters
batch_size=64
epochs=40
lr=0.001

transform=transforms.Compose([
                                   transforms.RandomRotation(10),
                                   transforms.Resize((50,50)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))

])


dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=transform)
train_data_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)


optimizer=torch.optim.Adam(model.parameters(),lr=lr)
loss_function=torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(epochs):
  train_loss=0.0
  train_acc=0.0
  for images,labels in tqdm(train_data_loader):
    images,labels=images.to(device),labels.to(device)
    optimizer.zero_grad()

    preds=model(images)
    loss=loss_function(preds,labels)
    loss.backward()
    optimizer.step()
    train_loss+=loss
    train_acc+=calc_acc(preds,labels)
  
  total_loss=train_loss/len(train_data_loader)
  total_acc=train_acc/len(train_data_loader)
  print(f"epochs: {epoch} , loss: {total_loss}, acc: {total_acc}")



torch.save(model.state_dict(),"persianMNist.pth")


