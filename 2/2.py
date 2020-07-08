#!/usr/bin/env python
# coding: utf-8

# Drawings on image

# In[12]:


import cv2
import numpy as np


# In[13]:


#img=cv2.imread("HappyFish.jpg",1)
img=np.zeros([512,512,3],np.uint8)


# In[14]:


img=cv2.line(img,(0,0),(100,100),(0,255,0),10)
img=cv2.arrowedLine(img,(0,100),(100,100),(255,0,0),10)
img=cv2.rectangle(img,(129,119),(159,179),(0,0,245),5)
img=cv2.circle(img,(99,159),23,(145,33,245),-1)
font=cv2.FONT_HERSHEY_SIMPLEX
img=cv2.putText(img,"Modem",(120,120),font,0.5,(193,5,155),1,cv2.LINE_AA)


# In[15]:


cv2.imshow("image",img)
cv2.imwrite("image3.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:




