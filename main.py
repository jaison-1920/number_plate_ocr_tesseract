import cv2
import pytesseract
import numpy as np

# used to set the path to the Tesseract OCR executable on a Windows system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#CascadeClassifier is a class within OpenCV that is used for object detection using Haar-like features.
# .xml file containing a pre-trained Haar cascade classifier model for detecting Russian number plates.
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

state_codes = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chattisgarh",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LD": "Lakshadweep Islands",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PY": "Pondicherry",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra & Nagar Haveli",
    "DD": "Daman & Diu",
    "LA": "Ladakh"
}

def extract_num(image):
    global read
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in nplate:
        a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))#img.shape[0] = height,img.shape[1]=width
        plate = img[y+a:y+h-a,x+b:x+w-b,:]
        kernel = np.ones((1,1),np.uint8)
        plate = cv2.dilate(plate,kernel,iterations=1)
        plate = cv2.erode(plate,kernel,iterations=1)
        plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate) = cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]

        try:
            print("The car belongs to",state_codes[stat])
        except:
            print('State is not recognized..!')
        print(read)

        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x,y-40),(x+w,y),(51,51,255),-1)
        cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1, 1, cv2.LINE_AA)
        cv2.imshow('Plate',plate)
    
    cv2.imshow('Result',img)
    cv2.imwrite('result.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

extract_num('./test_images/7.jpeg')




