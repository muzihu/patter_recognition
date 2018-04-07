import numpy as np
import os
import cv2

im = cv2.imread('./train/0.bmp',cv2.IMREAD_GRAYSCALE)
root_path = "~/GitBak/Pattern_Recognition/hk/hk1_2_3/train/"  
count = 0
for root,root_path,files in os.walk(root_path):  
    for file in files:  
        srcImg = cv2.imread(root_path+"images"+"/"+str(file))  
        roiImg = srcImg[36:521, 180:745]  
        cv2.imwrite(root_path+"Image"+"/"+str(file),roiImg)  
        count +=1  
        if count%400==0:  
            print count  

# import sys

# import numpy as np
# import cv2

# im = cv2.imread('digit.png')
# im3 = im.copy()

# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(5,5),0)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

# #################      Now finding Contours         ###################

# image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# samples =  np.empty((0,100))
# responses = []
# keys = [i for i in range(48,58)]

# for cnt in contours:
#     if cv2.contourArea(cnt)>50:
#         [x,y,w,h] = cv2.boundingRect(cnt)

#         if  h>28:
#             cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
#             roi = thresh[y:y+h,x:x+w]
#             roismall = cv2.resize(roi,(10,10))
#             cv2.imshow('norm',im)
#             key = cv2.waitKey(0)

#             if key == 27:  # (escape to quit)
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape((1,100))
#                 samples = np.append(samples,sample,0)

# responses = np.array(responses,np.float32)
# responses = responses.reshape((responses.size,1))
# print ("training complete")

# np.savetxt('generalsamples.data',samples)
# np.savetxt('generalresponses.data',responses)

print('Success')

# resize(a)
