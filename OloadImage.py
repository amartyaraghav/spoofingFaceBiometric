import cv2
import tkinter as tk
from tkinter import filedialog
import os
import random
import faceRecognition as fr
import numpy

def imageLoading():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    load_img=cv2.imread(file_path)
    return load_img

def imagePoisoning():
    path="/Users/raghav/Documents/Biometric Security/Poison on FaceRecognition/trainingImages"
    folder = ".DS_Store"
    while (folder == ".DS_Store"):
        files=os.listdir(path)
        folder=random.choice(files)
    print(folder)
    path = path+"/"+folder
    files=os.listdir(path)
    img =random.choice(files)
    path = path+"/"+img

    test_img=cv2.imread(path)
    return test_img

def poison(load_img):
    test_img=imagePoisoning()
    faces_detected,gray_img=fr.faceDetection(load_img)
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('trainingData.yml')
    name={0:"Priyanka",1:"Kangana",}#creating dictionary containing names for each label
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(load_img,face)
        predicted_name=name[label]
        if(confidence<100):#If confidence more than 100 then don't print predicted face text on screen
            fr.put_text(load_img,predicted_name,x,y)
        else:
            poison(test_img)
