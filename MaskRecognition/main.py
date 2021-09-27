from tensorflow.keras import models
from face_alignment import image_align
import sys
import os
import numpy as np

from tensorflow.keras.models import load_model


from PySide6.QtWidgets import QApplication, QWidget,QInputDialog,QLineEdit
from PySide6.QtGui import *
from PySide6 import QtGui
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QThread, Signal, QDir
from threading import Thread
import cv2
from PIL.ImageQt import ImageQt
from PIL import Image

from align_image import Align_image
image_aln=Align_image()


def convertCVImage2QtImage(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytesPerLine = 3 * width
    qimg = QImage(cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)



class Camera(QThread):
    camera_signal = Signal(object,int)
    def __init__(self):
        super(Camera, self).__init__()

    def run(self):
        self.video = cv2.VideoCapture(0)
        while True:
            valid, self.frame = self.video.read()
            if valid is not True:
                break
            pred=Process(self.frame)
            p=pred.predict()
            self.camera_signal.emit(self.frame,p)
            cv2.waitKey(1)

    def stop(self):
        try:
            self.video.release()
        except:
            pass



class Process(Thread):
    # pred_signal = Signal(object)
    def __init__(self,img):
        super(Process,self).__init__()
        self.img=img
        self.model=load_model('maskrec.h5')

    def predict(self):
        face=image_aln.aln(self.img)
        img_RGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img_RGB, (224, 224))
    
        img_numpy = np.array(img_resize)
        img = img_numpy / 255.0
        final = img.reshape(1, 224, 224, 3)

        y_pred = np.argmax(self.model.predict(final))
        # self.pred_signal.emit(y_pred)
        return y_pred






class GenderRecognition(QWidget):
    def __init__(self):
        super(GenderRecognition, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load('form.ui')
        self.ui.lbl_result.setVisible(False)
        self.ui.btn_start.clicked.connect(self.run_camera)
        self.thread_camera=Camera()
        self.thread_camera.camera_signal.connect(self.show_e)
        # self.thread_camera.camera_signal.connect(self.detect)

        # self.thread_camera.face_signal.connect(self.detect)
        self.ui.show()

    def show_e(self, img,p):
        res_img = convertCVImage2QtImage(img)
        self.ui.lbl_show.setPixmap(res_img)
        if p==0:
            self.ui.lbl_result.settext('mask')
        else:
            self.ui.lbl_result.settext('not mask')
        # self.detect(img)

    def run_camera(self):
        self.thread_camera.start()

    # def detect(self,img,p):
        

        

    

if __name__ == "__main__":
    app = QApplication([])
    widget = GenderRecognition()
    sys.exit(app.exec_())
