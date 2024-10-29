# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import Label, PhotoImage
from PIL import Image, ImageTk
from tkinter.tix import IMAGETEXT
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

window = tk.Tk()
window.title("Face Identification Window")

'''
bg = PhotoImage(file = "USF_Logo.png") 
  
# Show image using label 
label1 = Label( window, image = bg) 
label1.place(x = -1, y = -1)

window.configure(background='OliveDrab1')
'''
window.attributes('-fullscreen', True)


bg_image = Image.open("background_img.jpg")  # Replace with your image path
bg_image = bg_image.resize((window.winfo_screenwidth(),window.winfo_screenheight()), )  # Resize image if needed
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a canvas widget to place the background image
canvas = tk.Canvas(window, width=500, height=400)
canvas.pack(fill="both", expand=True)

# Set the background image on the canvas
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Attendance Marking Based on Face Identification", bg="green", fg="white", width=50, height=2, font=('Times', 30, 'bold underline')) 
message.place(x=350, y=20)

lbl = tk.Label(window, text="Provide ID Number:", width=20, height=2, fg="white"  ,bg="green", bd=2, highlightbackground="black", highlightthickness=1, font=('times', 15, ' bold ')) 
lbl.place(x=400, y=200)

txt = tk.Entry(window, width=20,  fg="black", bg="white", bd=2, highlightbackground="black", highlightthickness=1, font=('times', 15, ' bold '))
txt.place(x=600, y=205)

lbl2 = tk.Label(window, text="Provide your Name:", width=20, fg="white", bg="green", bd=2, highlightbackground="black", highlightthickness=1, height=2, font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=310)

txt2 = tk.Entry(window, width=20, fg="black", bg="white", bd=2, highlightbackground="black", highlightthickness=1, font=('times', 15, ' bold '))
txt2.place(x=600, y=315)

lbl3 = tk.Label(window, text="Current Status: ", width=20, fg="white", bg="green", bd=2, highlightbackground="black", highlightthickness=1 ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=400, y=420)

message = tk.Label(window, text="" , bg="green", bd=2, width=60  ,height=2, activebackground = "yellow" , highlightbackground="black", highlightthickness=1, font=('times', 15, ' bold ')) 
message.place(x=600, y=420)

lbl3 = tk.Label(window, text="Attendance: ", width = 20, fg="white", bg="green", height=2, highlightbackground="black", highlightthickness=1, font=('times', 15, ' bold  underline')) 
lbl3.place(x=400, y=650)

message2 = tk.Label(window, text="", fg="green", bg="white", activeforeground = "green", width=60, height=4, highlightbackground="black", highlightthickness=1, font=('times', 15, ' bold ')) 
message2.place(x=600, y=650)
 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        res = "Starting the Camera"
        message.configure(text= res)
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage/ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Captured Images for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails/StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Name must only include alphabets"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter ID in integers only"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Recognizer is Trained with Images"
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # Skip non-image files like .DS_Store
        if not imagePath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"Skipping non-image file: {imagePath}")
            continue
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            res = "Attendance Marked (Includes Time Stamp)"
            message.configure(text= res)
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)
    res=attendance
    message2.configure(text= res)

  
clearButton = tk.Button(window, text="Reset ID", command=clear  ,fg="red"  ,bg="yellow", relief="raised", width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=800, y=200)
clearButton2 = tk.Button(window, text="Reset Name", command=clear2  ,fg="red"  ,bg="yellow", relief="raised", width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=800, y=310)    
takeImg = tk.Button(window, text="1. Capture Images", command=TakeImages  ,fg="red"  ,bg="yellow", relief="raised", width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=250, y=500)
trainImg = tk.Button(window, text="2. Train Recognizer", command=TrainImages  ,fg="red"  ,bg="yellow", relief="raised", width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="3. Identify Person", command=TrackImages  ,fg="red"  ,bg="yellow", relief="raised", width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=750, y=500)
quitWindow = tk.Button(window, text="4. Quit Application", command=window.destroy  ,fg="red"  ,bg="yellow", relief="raised", width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1000, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 30, 'italic bold underline'))

# Use a Frame to position the Text widget at the bottom right
frame = tk.Frame(window)
frame.pack(side=tk.BOTTOM)#, anchor='se', padx=15, pady=15)

message3 = tk.Label(window,text="Group 9 Project",fg="black",bg="white",width="12",height="1",font=('times red', 16, ' bold'))
message3.place(x=1100, y=650)
 
window.mainloop()