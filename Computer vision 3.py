#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import datetime


# In[ ]:


cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap.set(3,720)
cap.set(4,720)
print(cap.get(3))
print(cap.get(4))


# In[ ]:


while(True):
    ret,frame=cap.read()
    font=cv2.FONT_HERSHEY_SIMPLEX
    date=str(datetime.datetime.now())
    img=cv2.putText(frame,date,(10,50),font,1,(0,255,255),2,cv2.LINE_AA)
    
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("video",frame)
    out.write(frame)
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
out.release()

cv2.destroyAllWindows()


# Mouse Events

# In[2]:


import numpy as np


# In[3]:


events=[i for i in dir(cv2) if "EVENT" in i ]


# In[4]:


print(events)


# In[5]:


def click_event(event,x,y,flag,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x," ",y)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strx=str(x)+" "+str(y)
        cv2.putText(img,strx,(x,y),font,0.5,(255,255,0),2)
        cv2.imshow("image",img)
    if event==cv2.EVENT_RBUTTONDOWN:
        blue=img[y,x,0]
        green=img[y,x,1]
        red=img[y,x,2]
        font=cv2.FONT_HERSHEY_SIMPLEX
        strx=str(blue)+" "+str(green)+" "+str(red)
        cv2.putText(img,strx,(x,y),font,0.5,(255,0,255),2)
        cv2.imshow("image",img)        

#img = np.zeros((512,512,3),np.uint8)     
img=cv2.imread("HappyFish.jpg")
cv2.imshow("image",img)
cv2.setMouseCallback("image",click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
    


# In[7]:


def click_event(event,x,y,flag,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        points.append((x,y))
        if len(points)>=2:
            cv2.line(img,points[-1],points[-2],(0,255,255),5)
        cv2.imshow("image",img)
points=[]        
img = np.zeros((512,512,3),np.uint8)     
#img=cv2.imread("HappyFish.jpg")
cv2.imshow("image",img)
cv2.setMouseCallback("image",click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[16]:


def click_event(event,x,y,flag,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        blue=img[x,y,0]
        green=img[x,y,1]
        red=img[x,y,2]
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        mycolorimage=np.zeros((512,512,3),np.uint8)
        mycolorimage[:]=[blue,green,red]
        cv2.imshow("color",mycolorimage)

points=[]        
#img = np.zeros((512,512,3),np.uint8)     
img=cv2.imread("HappyFish.jpg",1)
cv2.imshow("image",img)
cv2.setMouseCallback("image",click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


img=cv2.imread("messi.jpg",1)


# In[3]:


print(img.shape)


# In[6]:


print(img.size)


# In[7]:


print(img.dtype)


# In[8]:


b,g,r=cv2.split(img)


# In[9]:


b


# In[10]:


img=cv2.merge((b,g,r))


# In[11]:


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[15]:


ball=img[280:340,330:390]


# In[16]:


img[273:333,100:160]=ball


# In[17]:


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[19]:


img1=cv2.imread("HappyFish.jpg",1)


# In[20]:


img2=cv2.imread("messi.jpg",1)


# In[21]:


img1=cv2.resize(img1,(512,512))
img2=cv2.resize(img2,(512,512))


# In[22]:


dist=cv2.add(img1,img2)


# In[23]:


cv2.imshow("image",dist)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[28]:


dist=cv2.addWeighted(img1,0.3,img2,0.7,0)


# In[50]:


cv2.imshow("image",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


img1=np.zeros((250,500,3),np.uint8)
img1=cv2.rectangle(img1,(200,0),(300,100),(255,255,255),-1)


# In[4]:


img2=np.zeros((250,500,3),np.uint8)
img2=cv2.rectangle(img2,(100,20),(20,500),(255,255,255),-1)


# In[5]:


bitand=cv2.bitwise_and(img1,img2)


# In[6]:


cv2.imshow("image1",img1)
cv2.imshow("image2",img2)
cv2.imshow("bitand",bitand)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[7]:


bitor=cv2.bitwise_or(img1,img2)


# In[8]:


cv2.imshow("image1",img1)
cv2.imshow("image2",img2)
cv2.imshow("bitor",bitor)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


bitxor=cv2.bitwise_xor(img1,img2)


# In[10]:


cv2.imshow("image1",img1)
cv2.imshow("image2",img2)
cv2.imshow("bitxor",bitxor)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[11]:


bitnot1=cv2.bitwise_not(img1)
bitnot2=cv2.bitwise_not(img2)


# In[12]:


cv2.imshow("bitnot1",bitnot1)
cv2.imshow("bitnot2",bitnot2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Trackbar

# In[12]:


def nothing(x):
    print(x)
img=np.zeros((300,512,3),np.uint8)
cv2.namedWindow("image")
cv2.createTrackbar("B","image",0,255,nothing)
cv2.createTrackbar("G","image",0,255,nothing)
cv2.createTrackbar("R","image",0,255,nothing)
cv2.createTrackbar("Switch","image",0,1,nothing)
while(1):
    ret,frame=cap.read()
    
    k=cv2.waitKey(1)
    if k==27:
        break
    b=cv2.getTrackbarPos("B","image")    
    g=cv2.getTrackbarPos("G","image")    
    r=cv2.getTrackbarPos("R","image")
    s=cv2.getTrackbarPos("Switch","image")
    if s==0:
        img[:]=0
    else:    
        img[:]=[b,g,r]
cv2.destroyAllWindows()    
    


# In[1]:


import cv2
import numpy as np


# In[ ]:





# In[2]:


def nothing(x):
    print(x)
cv2.namedWindow("HSV")   
cv2.createTrackbar("LH","HSV",0,255,nothing)
cv2.createTrackbar("LS","HSV",0,255,nothing)
cv2.createTrackbar("LV","HSV",0,255,nothing)
cv2.createTrackbar("UH","HSV",0,255,nothing)
cv2.createTrackbar("US","HSV",0,255,nothing)
cv2.createTrackbar("UV","HSV",0,255,nothing)

while True:
    img=cv2.imread("messi.jpg",1)
    clr=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    LH=cv2.getTrackbarPos("LH","HSV")
    LS=cv2.getTrackbarPos("LS","HSV")
    LV=cv2.getTrackbarPos("LV","HSV")
    UH=cv2.getTrackbarPos("UH","HSV")
    US=cv2.getTrackbarPos("US","HSV")
    UV=cv2.getTrackbarPos("UV","HSV")
    lb=np.array([LH,LS,LV])
    ub=np.array([UH,US,UV])
    mask=cv2.inRange(clr,lb,ub)
    res=cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("imagee",img)
    cv2.imshow("image",res)
    cv2.imshow("mask",mask)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
    
 


# In[6]:


def nothing(x):
    print(x)
cv2.namedWindow("HSV")   
cv2.createTrackbar("LH","HSV",0,255,nothing)
cv2.createTrackbar("LS","HSV",0,255,nothing)
cv2.createTrackbar("LV","HSV",0,255,nothing)
cv2.createTrackbar("UH","HSV",0,255,nothing)
cv2.createTrackbar("US","HSV",0,255,nothing)
cv2.createTrackbar("UV","HSV",0,255,nothing)
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    clr=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow("video",frame)
    LH=cv2.getTrackbarPos("LH","HSV")
    LS=cv2.getTrackbarPos("LS","HSV")
    LV=cv2.getTrackbarPos("LV","HSV")
    UH=cv2.getTrackbarPos("UH","HSV")
    US=cv2.getTrackbarPos("US","HSV")
    UV=cv2.getTrackbarPos("UV","HSV")
    lb=np.array([LH,LS,LV])
    ub=np.array([UH,US,UV])
    mask=cv2.inRange(clr,lb,ub)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow("image",res)
    cv2.imshow("mask",mask)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()
 


# In[36]:


img=cv2.imread("HappyFish.jpg",0)


# In[48]:


_,thresh=cv2.threshold(img,33,255,cv2.THRESH_BINARY)
_,thresh1=cv2.threshold(img,33,255,cv2.THRESH_BINARY_INV)
_,thresh2=cv2.threshold(img,33,255,cv2.THRESH_TRUNC)
_,thresh3=cv2.threshold(img,33,255,cv2.THRESH_TOZERO)
_,thresh4=cv2.threshold(img,33,255,cv2.THRESH_TOZERO_INV)
_,thresh5=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,75,10);
cv2.imshow("image1",img)
cv2.imshow("image",thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[49]:


import matplotlib.pyplot as plt


# In[37]:





# In[38]:





# In[40]:





# In[ ]:




