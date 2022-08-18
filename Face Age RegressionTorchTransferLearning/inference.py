import argparse
import time
import torch
import torchvision
import cv2
import numpy as np


from align_image import align
import model


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--image", type=str, default='tests/test1.jpeg')
parser.add_argument("--kind", type=str, default='camera', help="image or camera?")
args = parser.parse_args()


transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0,0,0),(1,1,1))
])

device=args.device
m=model.Model()
m.load_state_dict(torch.load('weight.pth',map_location=args.device))
m.eval()



def pred(img, m):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = align(img)
    if np.all(face != None):
        img=cv2.resize(face,(32,32))
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
        y_pr=pred(frame,m)
    
        frame=cv2.putText(frame,str(y_pr), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 3, 2)
        print(y_pr)

        cv2.imshow('',frame)
        cv2.waitKey(1)

           
else:
    img=cv2.imread(args.image)
    o=pred(img,m)
    print(o)





