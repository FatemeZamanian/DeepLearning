import numpy as np
import cv2


from tensorflow.keras.models import load_model
from align_image import align


maskrec=load_model('Ages.h5')

def pred(img, model):
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = align(img)
    if np.all(face != None):
        img=cv2.resize(img,(224,224))
        img=img/255.0
        img=np.expand_dims(img,axis=0)
  
        age=model.predict(img)
        return age
    else:
        return -1


video = cv2.VideoCapture(0)
while True:
    valid, frame = video.read()
    if valid is not True:
        break


    y_pr=pred(frame,maskrec)
    
    frame=cv2.putText(frame,str(y_pr), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 3, 2)
    print(y_pr)

    cv2.imshow('',frame)
    cv2.waitKey(1)
