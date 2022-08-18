import torch
import argparse
from torchvision import transforms
import torchvision
import gdown
import model
from preprocess import test_set

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--dataset", type=str, default='cpu')
args = parser.parse_args()

device=torch.device(args.device)
device=torch.device(args.device if torch.cuda.is_available() else "cpu")

m=model.Model()
m.train(True)
m=m.to(device)

#hyper parameters
batch_size=64
epochs=5
lr=0.001

test_data_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)

m.load_state_dict(torch.load('weight.pth',map_location=device))
m.eval()

def calc_loss(pred,labels):
  # _,pred_max=torch.max(pred,1)
  loss_function=torch.nn.CrossEntropyLoss()
  loss=loss_function(pred,labels)
  return loss

loss=0
for images,labels in test_data_loader:
  images=images.to(device)
  labels=labels.to(device)

  preds=m(images)
  # print(pred,labels)
  loss+=calc_loss(preds,labels)
  
loss=loss/len(test_data_loader)
print(f"loss: {loss}")
