import cv2  
import numpy as np  
import matplotlib
import matplotlib.pyplot as plt
  
img_pre = cv2.imread("./test/1.bmp",0)  
img=cv2.adaptiveThreshold(img_pre,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,5)
img2 = img.copy()  
template_pre = cv2.imread("./train/1.bmp",0)  
ret,template = cv2.threshold(template_pre,template_pre.mean(),255,cv2.THRESH_BINARY)
w,h = template.shape[::-1]  
  
# 6 中匹配效果对比算法  
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',  
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']  
  
for meth in methods:  
    img = img2.copy()  

    method = eval(meth)  

    # 步长为1的扫描窗所得残差res 
    res = cv2.matchTemplate(img,template,method)  
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:  
        top_left = min_loc  
    else:  
        top_left = max_loc  
    bottom_right = (top_left[0] + w, top_left[1] + h)  

    # 通过确定对角线画矩形
    cv2.rectangle(img_pre,top_left, bottom_right, 255, 2)  

    print meth  
    plt.subplot(221), plt.imshow(img2,cmap= "gray")  
    plt.title('Original Image'), plt.xticks([]),plt.yticks([])  
    plt.subplot(222), plt.imshow(template,cmap= "gray")  
    plt.title('template Image'),plt.xticks([]),plt.yticks([])  
    plt.subplot(223), plt.imshow(res,cmap= "gray")  
    plt.title('Matching Result'), plt.xticks([]),plt.yticks([])  
    plt.subplot(224), plt.imshow(img_pre,cmap= "gray")  
    plt.title('Detected Point'),plt.xticks([]),plt.yticks([])  
    plt.show()  