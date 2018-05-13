import subprocess
import cv2
import numpy as np
import os ,shutil
from tkinter import Tk
from tkinter.filedialog import askopenfilename




def notEncloses(a1,b1,c1,d1,a2,b2,c2,d2):
    if c1*d1 > c2*d2:
        if a2>a1 and a2<a1+c1 and b2>b1 and b2<b1+d1 and a2+c2<a1+c1 and b2+d2<b1+d1:
            return 1
    return 0 

def no_alpha_at_last(plate_number):
    k=len(plate_number)
    i=k-1
    while (i>=0):
        if(plate_number[i].isalpha()):
            plate_number= plate_number[:i]
            i=i-1
        else:
            break
    return plate_number

def no_num_at_beg(plate_number):
    k=len(plate_number)
    i=0
    while (i<k):
        if(plate_number[i].isdigit()):
            plate_number= plate_number[1:]
        else:
            break
    return plate_number


if((os.path.isdir('char_result'))==False):
    os.mkdir('char_result')

if((os.path.isdir('plate_result'))==False):
    os.mkdir('plate_result')


Tk().withdraw() 
input_filename = askopenfilename() 
input_img=cv2.imread(input_filename)
flag=0
#cv2.imwrite("C:/full_code/testing/img.jpg",input_img)
path="testing/img.jpg"
p= subprocess.Popen(["alpr","-c","in","-j",input_filename], shell=True, stdout=subprocess.PIPE)
output = p.stdout.read()
out=output.decode("utf-8")
x1_in=999999
x2_in=-1
x3_in=-1
x4_in=999999
y1_in=999999
y2_in=999999
y3_in=-1
y4_in=-1
if out.find("coordinates")!=-1:
    coordinates=(out[out.find("coordinates"):out.find("candidates")])
        
    arr_x=[]
    arr_y=[]
    for i in range(len(coordinates)):
        if coordinates[i]=='x':
            arr_x.append(i)
    for i in range(len(coordinates)):
        if coordinates[i]=='y':
            arr_y.append(i)
                
    x1_in=coordinates[arr_x[0]+3:arr_y[0]-2]
    x2_in=coordinates[arr_x[1]+3:arr_y[1]-2]
    x3_in=coordinates[arr_x[2]+3:arr_y[2]-2]
    x4_in=coordinates[arr_x[3]+3:arr_y[3]-2]
    y1_in=coordinates[arr_y[0]+3:arr_x[1]-4]
    y2_in=coordinates[arr_y[1]+3:arr_x[2]-4]
    y3_in=coordinates[arr_y[2]+3:arr_x[3]-4]
    y4_in=coordinates[arr_y[3]+3:coordinates.find("]")-1]
else:
    flag=flag+1
        
p= subprocess.Popen(["alpr","-c","us","-j",input_filename], shell=True, stdout=subprocess.PIPE)
output = p.stdout.read()
out=output.decode("utf-8")
x1_us=999999
x2_us=-1
x3_us=-1
x4_us=999999
y1_us=999999
y2_us=999999
y3_us=-1
y4_us=-1
if out.find("coordinates")!=-1:
    coordinates=(out[out.find("coordinates"):out.find("candidates")])
    
    arr_x=[]
    arr_y=[]
    for i in range(len(coordinates)):
        if coordinates[i]=='x':
            arr_x.append(i)
            
    for i in range(len(coordinates)):
        if coordinates[i]=='y':
            arr_y.append(i)
   
    x1_us=coordinates[arr_x[0]+3:arr_y[0]-2]
    x2_us=coordinates[arr_x[1]+3:arr_y[1]-2]
    x3_us=coordinates[arr_x[2]+3:arr_y[2]-2]
    x4_us=coordinates[arr_x[3]+3:arr_y[3]-2]
    y1_us=coordinates[arr_y[0]+3:arr_x[1]-4]
    y2_us=coordinates[arr_y[1]+3:arr_x[2]-4]
    y3_us=coordinates[arr_y[2]+3:arr_x[3]-4]
    y4_us=coordinates[arr_y[3]+3:coordinates.find("]")-1]
else:
    flag=flag+1

p= subprocess.Popen(["alpr","-c","eu","-j",input_filename], shell=True, stdout=subprocess.PIPE)
output = p.stdout.read()
out=output.decode("utf-8")
x1_eu=999999
x2_eu=-1
x3_eu=-1
x4_eu=999999
y1_eu=999999
y2_eu=999999
y3_eu=-1
y4_eu=-1


if out.find("coordinates")!=-1:
    coordinates=(out[out.find("coordinates"):out.find("candidates")])
    
    arr_x=[]
    arr_y=[]
    for i in range(len(coordinates)):
        if coordinates[i]=='x':
            arr_x.append(i)
    for i in range(len(coordinates)):
        if coordinates[i]=='y':
            arr_y.append(i)
            
    x1_eu=coordinates[arr_x[0]+3:arr_y[0]-2]
    x2_eu=coordinates[arr_x[1]+3:arr_y[1]-2]
    x3_eu=coordinates[arr_x[2]+3:arr_y[2]-2]
    x4_eu=coordinates[arr_x[3]+3:arr_y[3]-2]
    y1_eu=coordinates[arr_y[0]+3:arr_x[1]-4]
    y2_eu=coordinates[arr_y[1]+3:arr_x[2]-4]
    y3_eu=coordinates[arr_y[2]+3:arr_x[3]-4]
    y4_eu=coordinates[arr_y[3]+3:coordinates.find("]")-1] 
else:
    flag=flag+1

if(flag==3):
    print("Plate not Found")

else:
    x1=min(int(x1_in),int(x1_us),int(x1_eu))
    x2=max(int(x2_in),int(x2_us),int(x2_eu))
    x3=max(int(x3_in),int(x3_us),int(x3_eu))
    x4=min(int(x4_in),int(x4_us),int(x4_eu))
    y1=min(int(y1_in),int(y1_us),int(y1_eu))
    y2=min(int(y2_in),int(y2_us),int(y2_eu))
    y3=max(int(y3_in),int(y3_us),int(y3_eu))
    y4=max(int(y4_in),int(y4_us),int(y4_eu))

    x1=int(x1-.35*(x2-x1))
    x4=int(x4-.35*(x2-x1))
    x2=int(x2+.35*(x2-x1))
    x3=int(x3+.35*(x2-x1))

    if(x1<0):
        x1=0

    if(x4<0):
        x4=0

    img_vehicle=cv2.imread(input_filename)
    max_r=len(img_vehicle[0])
    

    if(x2>max_r):
        x2=max_r-1

    if(x3>max_r):
        x3=max_r-1

    
    #cv2.imshow("Vehicle_image",img_vehicle)
    cv2.waitKey()
    yI=min(y1,y2)
    yF=max(y3,y4)
    xI=min(x1,x3)
    xF=max(x2,x4)
    img_license=img_vehicle[yI:yF,xI:xF]
    #cv2.imshow("LicensePlate",img_license)
    cv2.waitKey()
    #cv2.imwrite("C:/full_code/testing/license.jpg",img_license)
    img_gray=cv2.cvtColor( img_license, cv2.COLOR_BGR2GRAY );
    denoise=cv2.fastNlMeansDenoising(img_gray,None,10,7,21)
    binary_temp=cv2.adaptiveThreshold(denoise,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
    #cv2.imshow("Binary_image",binary_temp)
    cv2.waitKey()
    cv2.imwrite("plate_result/binary.jpg",binary_temp)
    binary_temp=cv2.imread("plate_result/binary.jpg")
    imgray = cv2.cvtColor(binary_temp, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 120, 255, cv2.THRESH_BINARY)

    cv2.imwrite("plate_result/binary.jpg",binary_temp)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("plate_result/thresh.jpg",imgray)
    
    binary_temp=cv2.imread("plate_result/binary.jpg")
    r,c,ch=binary_temp.shape
    plate_area=r*c
    arr=[]
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        x,y,w,h = cv2.boundingRect(contours[i])
        aspect_ratio = float(w)/h
        rect_area = w*h
        extent = float(area)/rect_area
        hull = cv2.convexHull(contours[i])
        hull_area = cv2.contourArea(hull)
        if hull_area>0:
          solidity = float(area)/hull_area
        else:
          solidity = 100  
        if area<=.04*plate_area and area>.004*plate_area:
            if aspect_ratio<3: #or extent<0.5 or extent>0.9 or solidity<0.3:
                 if h>.2*r and w<.2*c and h<.85*r:
                     arr.append(i)

    rej_arr=[]
    for i in arr:
        x1,y1,w1,h1=cv2.boundingRect(contours[i])
        for j in arr:
            if i!=j:
                x2,y2,w2,h2=cv2.boundingRect(contours[j])
                if notEncloses(x1,y1,w1,h1,x2,y2,w2,h2)==1:
                    rej_arr.append(j)
    for i in rej_arr:
        for j in arr:
            if i==j:
                arr.remove(i)        
    hash_arr = [[0 for a in range(2)] for b in range(len(arr))]    
    for s in range(0,len(arr)):
        x,y,w,h=cv2.boundingRect(contours[arr[s]])
        hash_arr[s][0]=x
        hash_arr[s][1]=arr[s]
    hash_arr.sort()


    folder = 'char_result/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


    k=11
    
    for i in range(0,len(arr)):
        x,y,w,h = cv2.boundingRect(contours[hash_arr[i][1]])
        img_path="char_result/char-"+str(k)+".jpg"
        im=binary_temp[y:y+h,x:x+w]
        cv2.imwrite(img_path,im)
        #plate_number+=(chal_ja.reformat1(img_path))
        k=k+1
        #cv2.rectangle(binary_temp,(x,y),(x+w,y+h),(0,255,0),1)
    #cv2.imshow("Character Segments",binary_temp)
    cv2.waitKey(0)
    import ocr
    plate_number=ocr.reformat1()
    print("original detected "+plate_number)
    plate_number=no_alpha_at_last(plate_number)
    plate_number=no_num_at_beg(plate_number)
    print("after formatting "+plate_number)




