import argparse
import wandb
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm


from preprocess import train_set
from model import *



parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
# parser.add_argument("--dataset", type=str,help='directory of dataset')
args = parser.parse_args()


device=torch.device(args.device)

device=torch.device(args.device if torch.cuda.is_available() else "cpu")
model=Model().to(device)

model.train(True)

#hyper parameters
batch_size=64
epochs=5
lr=0.001

train_data_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size)


#compile 
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
loss_function=nn.L1Loss()

model.train()
for epoch in range(epochs):

  train_loss=0.0
  for images,labels in tqdm(train_data_loader):
    images,labels=images.to(device),labels.to(device)
    optimizer.zero_grad()

    preds=model(images)
    ls=loss_function(preds,labels.float())
    ls.backward()
    optimizer.step()
    train_loss+=ls
  
  total_loss=train_loss/len(train_data_loader)
  print(f"epochs: {epoch} , loss: {total_loss}")

torch.save(model.state_dict(),"weight.pth")


