#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2


# In[ ]:


img=cv2.imread("HappyFish.jpg",0)


# In[ ]:


cv2.imshow("image",img)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
elif k==ord("s"):
    cv2.imwrite("1.jpg",img)


# In[7]:


cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))


# In[8]:


while(True):
    ret,frame=cap.read()
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out.write(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("video",gray)
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
out.release()

cv2.destroyAllWindows()


# In[ ]:




