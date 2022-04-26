import argparse
import time
import torch
import torchvision
import cv2
import numpy as np

import model


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--image", type=str, default='test1.png')
args = parser.parse_args()


transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0),(1)),
])

device=args.device
m=model.MyModel()
m.load_state_dict(torch.load('weight.pth',map_location=args.device))
m.eval()
img=cv2.imread(args.image)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(28,28))
tensor=transform(img).unsqueeze(0)
tensor=tensor.to(device)
start=time.time()
pred=m(tensor)
print(f"{time.time()-start} sec")
# pred=pred.argmax()
pred=pred.cpu().detach().numpy()
out=np.argmax(pred)
print(out)


