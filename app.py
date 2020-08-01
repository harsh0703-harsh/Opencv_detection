import cv2
import numpy as np
import streamlit as st
from imutils.video import WebcamVideoStream
from PIL import ImageColor,Image
import os
import io
import imutils




def main():
    colorr=""" <style>
     body{
     background-color:rgb(209, 228, 37);
     }
   
    </style> """
    st.markdown(colorr,unsafe_allow_html=True)
    option=["About","Images Detections","Camera"]
    choices=st.sidebar.selectbox("Choices",option)
    if choices=="About":
        st.title("Welcome to Web OpenCV applications")
        st.header("Please select the images Detections or Camera from sidebar options")
        st.title("Please follow the steps ")
        st.header("For Images Detections Tab")
        st.write("1. First select the Images Detections tab from sidebar selectbox")
        st.write("2. You can select the formats like .png , jpg , jpeg ")
        st.write("3. After that you can do eye detection , car detection and face detection in it")

        st.header("For Camera Detections")
        st.write("First you will se the Normal and Grey buttons at sidebar ")
        st.write("If you select the normal then it turn on your webcam at normal and then press Start button")
        st.write("If you selected the Grey button it will open your webcam in grey mode")

    if choices=="Images Detections":
        st.title("Images Detection")

        type_of_files=['png','jpg','jpeg']
        st.set_option('deprecation.showfileUploaderEncoding', False)
        images=st.file_uploader("Upload",type=type_of_files)
        if st.button("Show"):
            if images is not None:
                to_this=Image.open(images)
                st.image(to_this)
    elif choices=="Camera":
        st.title("Welcome to  detection")
        st.info("Please Start Button after work done")
        st.sidebar.title("Select the modes")
        if st.button("Start Button"):
            st.warning("Go to the sidebar options")
        if st.sidebar.button("Grey"):
            cap=cv2.VideoCapture(0)
            img=st.empty()
            st.warning("Please Press start button again to stop this")
            while True:
                cond,frames=cap.read()
                grey=cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
                frames=cv2.flip(frames,1)
                frames=imutils.resize(frames,width=700)
                img.image(grey,channels="RGB")
               #cv2.imshow("Grey Mode",grey)
                if cv2.waitKey(1) & 0xFF==ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
           
        if st.sidebar.button("Normal"):
            cap=cv2.VideoCapture(0)
            img=st.empty()
            while True:
                cond,frames=cap.read()
                frames=cv2.flip(frames,1)
                frames=imutils.resize(frames,width=700)
                img.image(frames,channels="BGR")
               #cv2.imshow("camera Frame",frames)
                if cv2.waitKey(1) & 0xFF==ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
        types_detect=["Select",'Eye & Face Detections','Full body Detections','Car Detections'] 
        func= st.sidebar.selectbox("Select the Detections ",types_detect)
        face_file = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_file = cv2.CascadeClassifier('haarcascade_eye.xml')
        body=cv2.CascadeClassifier("haarcascade_fullbody.xml")
        car_file=cv2.CascadeClassifier("haarcascade_car.xml")
        if func=="Eye & Face Detections":
            st.title("Face and Eye Detection")
            if st.button("Start"):
                def detecting(gray,frame):
                    faces=face_file.detectMultiScale(gray,1.3,3)
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,321,0),2)
                        gray_1=gray[y:y+h,x:x+w]
                        color_1=frame[y:y+h,x:x+w]
                        eyes=eye_file.detectMultiScale(gray_1,1.3)
                        for (e_x,e_y,e_w,e_h) in eyes:
                            img=st.empty()
                            cv2.rectangle(color_1,(e_x,e_y),(e_x+e_w,e_y+e_h),(213,524,0),4)
                    return frame
                cap=cv2.VideoCapture(0)
                img=st.empty()
                while True:
                    ret,frame=cap.read()
                    frame=cv2.flip(frame,1)
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    frame=imutils.resize(frame,width=700)
                    y=detecting(gray,frame)
                    img.image(y,channels="BGR")
                    if cv2.waitKey(1) & 0xFF==ord("q"):
                        break
                cap.release()
                cv2.destroyAllWindows()
        elif func=="Full body Detections":
            st.title("Human Detections")
            if st.button("Start Recognisation"):
                cap=cv2.VideoCapture("walking.avi")  # You can provide 0 for your webcam.
                img=st.empty()
                while True:
                    ret,frame=cap.read()
                    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    frame=imutils.resize(frame,width=700)
                    HUMAN=body.detectMultiScale(grey,1.1,2)
                    for (x,y,w,h) in HUMAN:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,245),3)
                    img.image(frame,channels='BGR')
                    if cv2.waitKey(1) & 0xFF==ord("q"):
                        break
                cap.release()
                cv2.destroyAllWindows()
        elif func=='Car Detections':
            st.title("Car Detections")
            if st.button("Start"):
                cap=cv2.VideoCapture(0)
                img=st.empty()
                while True:
                    ret,frame=cap.read()
            #        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    frame=cv2.flip(frame,1)
                    frame=imutils.resize(frame,width=800)
                    cars=car_file.detectMultiScale(frame,1.3,2)
                    for (x,y,w,h) in cars:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,265),4)
                    img.image(frame,channels="BGR")
                    if cv2.waitKey(1) & 0xFF==ord("q"):
                        break
                cap.release()
                cv2.destroyAllWindows()

                

    





@st.cache
def images(img):
    ima=Image.open(img)
    return ima
        

if __name__ =="__main__":
    main() 


