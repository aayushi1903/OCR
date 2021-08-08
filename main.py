#importing necessary modules
import numpy as np
import cv2
import os,io
import csv
from csv import writer
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import glob
from PIL import Image
import sys

#defining necessary paths 

FOLDER_PATH = r'C:/Users/aayus/Downloads/AI Systems Final/VisionAPIDemo/Images'
IMG_DIR = 'Images/'

# initializing cloudvisionAPI
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'VisionAPI.json'
client=vision_v1.ImageAnnotatorClient()



#defining ROI's
roi_A = [[(40,57),(670,99),'text','Name'],
        [(703,125),(810,162),'text','Age'],
        [(47,129),(322,163),'text','Date'],
        [(333,130),(442,157),'text','Section'],
        [(616,122),(698,159),'text','Gr'],
        [(146,183),(969,215),'text','Undertaker'],
        [(850,119),(983,158),'text','Veteran']]

roi_B = [[(105,5),(986,60),'text','Name'],
        [(85,229),(722,259),'text','Age'],
        [(184,176),(920,221),'text','Date'],
        [(578,78),(972,111),'text','Section'],
        [(80,78),(470,111),'text','Lot']]

roi_C= [[(78,8),(907,86),'text','Name'],
        [(58,233),(716,275),'text','Age'],
        [(151,179),(675,230),'text','Date'],
        [(546,84),(919,125),'text','Section'],
        [(77,85),(392,133),'text','Lot']]

#functions to preprocess the image
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


#defining function to read data via CloudVision API
def read_data(file_name):
    IMAGE_FILE=file_name
    FILE_PATH = os.path.join(FOLDER_PATH, IMAGE_FILE)
    with io.open(FILE_PATH, 'rb') as image_file:
        content = image_file.read()
    image = vision_v1.types.Image(content=content)
    response = client.document_text_detection(image=image)
    text = response.full_text_annotation.text
    return(text)

#function to save cropped images
def save_image(image,image_name):
    filename=os.path.join(FOLDER_PATH,image_name)
    cv2.imwrite(filename,image)

#function to extract data and save into csv file
def extract_data(roi,empty_filename,images_path,file_csv):
    imgQ = cv2.imread(IMG_DIR +empty_filename,0)
    h,w = imgQ.shape
    #imgQ = cv2.resize(imgQ,(w // 2, h // 2))

    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(imgQ,None)
    #imgkp1 = cv2.drawKeypoints(imgQ,kp1,None)

    path = images_path
    myPicList = os.listdir(path)
    print(myPicList)
    for j,y in enumerate(myPicList):
        img = cv2.imread(path + "/"+y)
        #img = cv2.resize(img,(w // 2, h // 2))
        #cv2.imshow(y, img)
        kp2 , des2 = orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des2,des1)
        matches.sort(key= lambda x: x.distance)
        good = matches[:100]
        imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good,None,flags=2)
        #cv2.imshow(y, imgMatch)
        
        srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dstPoints = np.float32([kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M , _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
        imgScan = cv2.warpPerspective(img,M,(w,h))
        #imgScan = cv2.resize(imgScan,(w // 2, h // 2))
        #cv2.imshow(y, imgScan)
        #cv2.waitKey(0)

        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)
        
        myData = []
        
        for x,r in enumerate(roi):
            cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
            imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
        
            imgCrop = imgScan[r[0][1]:r[1][1],r[0][0]:r[1][0]]
            save_image(imgCrop,"imgCrop.jpg")
            #cv2.imshow(str(x), imgCrop)
            #cv2.waitKey(0)
            
            test=read_data("imgCrop.jpg")
            myData.append(test)
            print(myData)
            
        #writing data into the csv file
        with open(file_csv, 'a') as f_object:

            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)

            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(myData)

            #Close the file object
            f_object.close()


#main
def main():
    
    print('''Please select one of the options to proceed:
              1. Read and extraxt data from Type A records
              2. Read and extract data from Type B records
              3. To Exit''')
    strInput = input("Enter any of the above options to proceed: ")

    if strInput=="1":
        extract_data(roi_A,'Empty_A.jpg','resized_A','type_A.csv')
        print("-----------Succesfully extracted and stored the data!-----------------")
        main()

    elif strInput=="2":
        extract_data(roi_B,'Empty_B.jpg','resized_B','type_B.csv')
        extract_data(roi_C,'Empty_C.jpg','resized_C','type_B.csv')
        print("-----------------Succesfully extracted and stored the data!----------------------")
        main()

    elif strInput == '3':
        print("Exiting!!!")
        sys.exit("Exited")

    else:
        print("Oops!You have entered an invalid option!\nPlease try again \n")
        main()

if __name__=="__main__":
    #Resizing the images

    #Defining new width and height of image
    new_width  = 1000
    new_height = 600

    #Resizing type A images
    imageNames = glob.glob(r"C:\Users\aayus\Downloads\AI Systems Final\VisionAPIDemo\input_A\*.jpg")
     #Count variable to show the progress of image resized
    count=0

    #Creating for loop to take one image from imageNames list and resize
    for i in imageNames:
        #opening image for editing
        img = Image.open(i)
        #using resize() to resize image
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        #save() to save image at given path and count is the name of image eg. first image name will be 0.jpg
        img.save(r"C:\Users\aayus\Downloads\AI Systems Final\VisionAPIDemo\resized_A\\"+str(count)+".jpg") 
        #incrementing count value
        count+=1
        #showing image resize progress
        #print("Images Resized " +str(count)+"/"+str(len(imageNames)),end='\r')
    
    #Resizing type B images
    imageNames = glob.glob(r"C:\Users\aayus\Downloads\AI Systems Final\VisionAPIDemo\input_B\*.jpg")
    #Count variable to show the progress of image resized
    count=0

    #Creating for loop to take one image from imageNames list and resize
    for i in imageNames:
        #opening image for editing
        img = Image.open(i)
        #using resize() to resize image
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        #save() to save image at given path and count is the name of image eg. first image name will be 0.jpg
        img.save(r"C:\Users\aayus\Downloads\AI Systems Final\VisionAPIDemo\resized_B\\"+str(count)+".jpg") 
        #incrementing count value
        count+=1
        #showing image resize progress
        #print("Images Resized " +str(count)+"/"+str(len(imageNames)),end='\r')

    #Resizing type C images
    imageNames = glob.glob(r"C:\Users\aayus\Downloads\AI Systems Final\VisionAPIDemo\input_C\*.jpg")
    #Count variable to show the progress of image resized
    count=0

    #Creating for loop to take one image from imageNames list and resize
    for i in imageNames:
        #opening image for editing
        img = Image.open(i)
        #using resize() to resize image
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        #save() to save image at given path and count is the name of image eg. first image name will be 0.jpg
        img.save(r"C:\Users\aayus\Downloads\AI Systems Final\VisionAPIDemo\resized_C\\"+str(count)+".jpg") 
        #incrementing count value
        count+=1
        #showing image resize progress
        #print("Images Resized " +str(count)+"/"+str(len(imageNames)),end='\r')

    main()