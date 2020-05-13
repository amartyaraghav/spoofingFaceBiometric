import cv2
import OloadImage as li 
import faceRecognition as fr
import os
import numpy

#def faceDetection():
load_img = li.imageLoading()
test_img = li.imagePoisoning()
li.poison(load_img)

#Comment below lines for subsequent running (We do not need to train our model everytime.)
#faces,faceID=fr.labels_for_training_data('trainingImages')
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.write('trainingData.yml')

resized_img=cv2.resize(load_img,(1000,1000))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows