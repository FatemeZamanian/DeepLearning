import torch
import argparse
from torchvision import transforms
import torchvision
import gdown
import model

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
batch_size=16
epochs=40
lr=0.0001

transform=transforms.Compose([
                                   transforms.RandomRotation(10),
                                   transforms.Resize((50,50)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))

])

# url="https://drive.google.com/drive/folders/1en-zjfGWNaGTuJ0_AK-TnT9YL3Lfs5_J?usp=sharing"
# out="pmnist"
# gdown.download_folder(url)

dataset=torchvision.datasets.ImageFolder(root=args.dataset, transform=transform)
test_data_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

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
