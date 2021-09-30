import numpy as np
import cv2


from tensorflow.keras.models import load_model
from align_image import align


maskrec=load_model('gender.h5')

def pred(img, model):
    # img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = align(img)
    if np.all(face != None):
        img_resize = cv2.resize(face, (224, 224))
        img_numpy = np.array(img_resize)
        img = img_numpy / 255.0
        final = img.reshape(1, 224, 224, 3)

        y_pred = np.argmax(model.predict(final))
        return y_pred
    else:
        return -1


video = cv2.VideoCapture(0)
while True:
    valid, frame = video.read()
    if valid is not True:
        break


    y_pr=pred(frame,maskrec)
    if y_pr==0:
        res='Gender : Female'
    elif y_pr==1:
        res='Gender : Male'
    else:
        res='not found face'
    frame=cv2.putText(frame,res, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 3, 2)
    print(y_pr)

    cv2.imshow('',frame)
    cv2.waitKey(1)
