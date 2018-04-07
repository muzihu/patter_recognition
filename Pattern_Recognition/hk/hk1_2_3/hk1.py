import cv2  
import numpy as np  
import matplotlib
import matplotlib.pyplot as plt
  

img_pre = cv2.imread("./test/噪声.bmp",0)  
img=cv2.adaptiveThreshold(img_pre,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,17)
img2 = img.copy()  

meth = 'cv2.TM_CCOEFF'
print meth  
plt.subplot(121), plt.imshow(img_pre,cmap= "gray")  
plt.title('Original Image'), plt.xticks([]),plt.yticks([])  
plt.subplot(122), plt.imshow(img,cmap= "gray")  
plt.title('Original Image'), plt.xticks([]),plt.yticks([])  
plt.show() 
cv2.waitKey(1)

template_num = ["0","1","2","3","4","6","8","9"]
w = np.zeros((8,1))
h = np.zeros((8,1))

img = img2.copy()  
method = eval(meth)  
template_pre = {}
res = {}

for i in range(8):
    template_pre_single = cv2.imread("./train/"+template_num[i]+".bmp",0)  
    ret_single,template_single = cv2.threshold(template_pre_single,template_pre_single.mean(),255,cv2.THRESH_BINARY)
    w[i],h[i] = template_single.shape[::-1]  
    template_pre[i] = template_single

for i in range(8): 
    tmp1 = int((max(w)-w[i])/2)
    tmp2 = int(max(w)-w[i]-tmp1)
    tmp3 = int((max(h)-h[i])/2)
    tmp4 = int(max(h)-h[i]-tmp3)
    template_pre[i] = cv2.copyMakeBorder(template_pre[i], tmp3,tmp4,tmp1,tmp2, cv2.BORDER_CONSTANT, value=[255,255,255])
    # 步长为1的扫描窗所得残差res 
    res_tmp = cv2.matchTemplate(img,template_pre[i],method) 
    res[i] = res_tmp

    # print meth  
    # plt.subplot(111), plt.imshow(template_pre[i],cmap= "gray")  
    # plt.title('Original Image'), plt.xticks([]),plt.yticks([])  
    # plt.show() 
    # v2.waitKey(1)

res_max = np.zeros_like(res_tmp)
num_like = np.zeros_like(res_tmp)
for i in range(len(res_tmp)):
    for j in range(len(res_tmp[0])):
        res_max_tmp = [res[0][i][j],res[1][i][j],res[2][i][j],res[3][i][j],res[4][i][j],res[5][i][j],res[6][i][j],res[7][i][j]]
        res_max[i][j] = max(res_max_tmp)
        num_like[i][j] = res_max_tmp.index(max(res_max_tmp)) 

cho_max = 14
top_left = {}
cho_num = np.zeros((cho_max,1))
for i in range(cho_max):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_max)  
    top_left[i] = max_loc  
    bottom_right = (max_loc[0] + int(max(w)), max_loc[1] + int(max(h)))
    # 通过确定对角线画矩形
    cv2.rectangle(img_pre,max_loc, bottom_right, 255, 2) 
    cho_num[i] = num_like[max_loc[1]][max_loc[0]] 
    for j in range(max(w)/2-1):
        for k in range(max(h)/2-1):
            if ((max_loc[1]+k)<len(res_max) and (max_loc[0]+j)<len(res_max[0])) :
                res_max[max_loc[1]+k][max_loc[0]+j] = 0
            if ((max_loc[1]-k)>=0 and (max_loc[0]-j)>=0) :
                res_max[max_loc[1]-k][max_loc[0]-j] = 0
            if ((max_loc[1]+k)<len(res_max) and (max_loc[0]-j)>=0) :
                res_max[max_loc[1]+k][max_loc[0]-j] = 0
            if ((max_loc[1]-k)>=0 and (max_loc[0]+j)<len(res_max[0])) :
                res_max[max_loc[1]-k][max_loc[0]+j] = 0
    # print(cho_num[i])
    print('识别的第%d个数字是%s，位置在%d， %d' %(i+1,template_num[int(cho_num[i])],max_loc[0],max_loc[1]))
print(max_val)

print meth  
plt.subplot(221), plt.imshow(img2,cmap= "gray")  
plt.title('Original Image'), plt.xticks([]),plt.yticks([])  
plt.subplot(222), plt.imshow(template_pre_single,cmap= "gray")  
plt.title('template Image'),plt.xticks([]),plt.yticks([])  
plt.subplot(223), plt.imshow(res_max,cmap= "gray")  
plt.title('Matching Result'), plt.xticks([]),plt.yticks([])  
plt.subplot(224), plt.imshow(img_pre,cmap= "gray")  
plt.title('Detected Point'),plt.xticks([]),plt.yticks([])  
plt.show()  

    

print("success")