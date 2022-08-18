import argparse
import time
import torch
import cv2
import numpy as np
from torchvision import transforms
from align_image import align
import model
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--image", type=str, default='tests/test2.jpg')
parser.add_argument("--kind", type=str, default='image', help="image or camera?")
args = parser.parse_args()


transform = transforms.Compose([
    # transforms.PILToTensor(),
    transforms.Resize((160,160)),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    # torchvision.transforms.Normalize((0), (1))
])


device=args.device
m=model.Model()
m.load_state_dict(torch.load('weight.pth',map_location=args.device))
m.eval()

def pred(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = align(img)
    if np.all(face != None):
        img=cv2.resize(face,(160,160))
        img = Image.fromarray(img)
        tensor=transform(img).unsqueeze(0)
        tensor=tensor.to(device)
        start=time.time()
        pred=m(tensor)
        print(f"{time.time()-start} sec")
        pred=pred.cpu().detach().numpy()
        return pred
    else:
        return -1




if args.kind =="camera":
    cap=cv2.VideoCapture(0)
    while True:
            valid, frame = cap.read()
            if valid is not True:
                break

            row, col, ch = frame.shape
            mask = frame[row //2-100:row // 2+100, col // 2-100:col // 2+100]
            frame=cv2.medianBlur(frame,33)
            frame[0:row]=(255,0,0)
            frame[row // 2-100:row // 2+100, col // 2-100:col // 2+100] = mask
            y_pr=pred(mask)
            frame=cv2.putText(frame,str(y_pr), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 3, 2)
            cv2.imshow('',frame)
            cv2.waitKey(1)

else:
    img=cv2.imread(args.image)
    o=pred(img)
    print(o)





