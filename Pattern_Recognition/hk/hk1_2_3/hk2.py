'''

运行环境：python 2.7 
运行需要的库文件：openCV， numpy， matplotlib
运行方法： 运行python hk2.py。（test和train文件夹放在hk1.py当前目录下即可）

'''

import cv2  
import numpy as np  
import matplotlib.pyplot as plt

def haar_cal(img_1, img_2):
    num1 = sum(sum(img_1))
    num2 = sum(sum(img_2))
    return num1 - num2

def div_pic(img, x, y, w, h):
    cut_img = img[x:x+h, y:y+w]
    return np.array(cut_img)

def get_fea(img):
    feature = []
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    ret_single,img = cv2.threshold(img,img.mean(),255,cv2.THRESH_BINARY)

    # plt.subplot(111), plt.imshow(img,cmap= "gray")  
    # plt.show()  

    # 以2×2为一个方块抽取哈尔特征：
    for i in range(0, 32, 4):
        for j in range(0, 32, 2):
            split_img = np.array(img[i:i+4, j:j+2])
            split_img[split_img == 255] = 1
            num = haar_cal(np.array(split_img[:2,:]), np.array(split_img[2:,:]))
            feature.append(num)
    # 以2×4为一个方块提取哈尔特征：
    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            split_img = np.array(img[i:i+4, j:j+4])
            split_img[split_img == 255] = 1
            num = haar_cal(split_img[:2,:], split_img[2:,:])
            feature.append(num)
    # 以4×4为一个方块提取哈尔特征：
    for i in range(0, 32, 8):
        for j in range(0, 32, 4):
            split_img = np.array(img[i:i+8, j:j+4])
            split_img[split_img == 255] = 1
            num = haar_cal(split_img[:4,:], split_img[4:,:])
            feature.append(num)
    # 以8×16为一个方块提取哈尔特征：
    for i in range(0, 32, 16):
        for j in range(0, 32, 16):
            split_img = np.array(img[i:i+16, j:j+16])
            split_img[split_img == 255] = 1
            num = haar_cal(split_img[:8, :], split_img[8:, :])
            feature.append(num)
    # 以16×16为一个方块提取哈尔特征：
    for i in range(0, 32, 32):
        for j in range(0, 32, 16):
            split_img = np.array(img[i:i+32, j:j+16])
            split_img[split_img == 255] = 1
            num = haar_cal(split_img[:16, :], split_img[16:, :])
            feature.append(num)
    return feature

def matchTemplate(img,mod_haar,size):
    res = np.ones((len(img)-int(size[0]),len(img[0])-int(size[1])))*1000000
    for i in range(len(res)):
        for j in range(len(res[0])):
            # if i%2==0 and j%2==0 :
            img_tmp = img[i:i+int(size[0]),j:j+int(size[1])]
            sample_haar = get_fea(img_tmp)
            res[i][j] = sum((np.array(sample_haar) - np.array(mod_haar)) ** 2)
    return res

if __name__ == '__main__':
    test_num = ["1","2","3","4","5","6","划痕","噪声"]  
    for test_test in range(8): 
        img_pre = cv2.imread("./test/"+test_num[test_test]+".bmp",0)
        img_pre2 = img_pre.copy()    
        img=cv2.adaptiveThreshold(img_pre,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,5)
        # ADAPTIVE_THRESH_GAUSSIAN_C
        img2 = img.copy()  
        template_num = ["0","1","2","3","4","6","8","9"]
        w = np.zeros((8,1))
        h = np.zeros((8,1))

        img = img2.copy()  
        template_pre = {}
        mod_haar = {}
        res = {}

        for i in range(8):
            template_pre[i] = cv2.imread("./train/"+template_num[i]+".bmp",0)  
            w[i],h[i] = template_pre[i].shape[::-1]  

        for i in range(8): 
            tmp1 = int((max(w)-w[i])/2)
            tmp2 = int(max(w)-w[i]-tmp1)
            tmp3 = int((max(h)-h[i])/2)
            tmp4 = int(max(h)-h[i]-tmp3)

            ret_single,template_pre[i] = cv2.threshold(template_pre[i],template_pre[i].mean(),255,cv2.THRESH_BINARY)
            template_pre[i] = cv2.copyMakeBorder(template_pre[i], tmp3,tmp4,tmp1,tmp2, cv2.BORDER_CONSTANT, value=[255,255,255])
            template_pre[i] = cv2.resize(template_pre[i], (32, 32), interpolation=cv2.INTER_CUBIC)
            ret_single,template_pre[i] = cv2.threshold(template_pre[i],template_pre[i].mean(),255,cv2.THRESH_BINARY)
            # size = [32, 32]

            size = [max(h), max(w)]

            mod_haar[i] = get_fea(template_pre[i])
            # 步长为1的扫描窗所得残差res 
            res_tmp = matchTemplate(img,mod_haar[i],size) 
            res[i] = res_tmp
 

        res_min = np.zeros_like(res_tmp)
        num_like = np.zeros_like(res_tmp)
        for i in range(len(res_tmp)):
            for j in range(len(res_tmp[0])):
                res_min_tmp = [res[0][i][j],res[1][i][j],res[2][i][j],res[3][i][j],res[4][i][j],res[5][i][j],res[6][i][j],res[7][i][j]]
                res_min[i][j] = min(res_min_tmp)
                num_like[i][j] = res_min_tmp.index(min(res_min_tmp)) 

        if test_test<6:
            threshold_i = 7e6
        elif test_test==6:
            threshold_i = 6e6
        elif test_test==7:
            threshold_i = 4.3e6
        else:
            break
        cho_max = 20
        top_left = {}
        cho_num = {}
        i = 0
        # while True:
        for i in range(cho_max):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_min)  
            # if min_val>threshold_i:
            #     break
            top_left[i] = min_loc  
            bottom_right = (min_loc[0] + int(max(w)), min_loc[1] + int(max(h)))
            # 通过确定对角线画矩形
            cv2.rectangle(img_pre,min_loc, bottom_right, 255, 2) 
            cho_num[i] = num_like[min_loc[1]][min_loc[0]] 
            img_save = img_pre2[min_loc[1]:min_loc[1]+int(max(h)),min_loc[0]:min_loc[0]+int(max(w))]
            for j in range(max(w)/2):
                for k in range(max(h)/2):
                    if ((min_loc[1]+k)<len(res_min) and (min_loc[0]+j)<len(res_min[0])) :
                        res_min[min_loc[1]+k][min_loc[0]+j] = 1000000
                    if ((min_loc[1]-k)>=0 and (min_loc[0]-j)>=0) :
                        res_min[min_loc[1]-k][min_loc[0]-j] = 1000000
                    if ((min_loc[1]+k)<len(res_min) and (min_loc[0]-j)>=0) :
                        res_min[min_loc[1]+k][min_loc[0]-j] = 1000000
                    if ((min_loc[1]-k)>=0 and (min_loc[0]+j)<len(res_min[0])) :
                        res_min[min_loc[1]-k][min_loc[0]+j] = 1000000
            # print(cho_num[i])
            print('识别的第%d个数字是%s，位置在%d， %d' %(i+1,template_num[int(cho_num[i])],min_loc[0],min_loc[1]))
            cv2.imwrite('./'+test_num[test_test]+'/'+str(i)+'_'+template_num[int(cho_num[i])]+'.jpg',img_save)
            # i += 1
        print(min_val)

        plt.subplot(221), plt.imshow(img2,cmap= "gray")  
        plt.title('Original Image'), plt.xticks([]),plt.yticks([])  
        plt.subplot(223), plt.imshow(res_min,cmap= "gray")  
        plt.title('Matching Result'), plt.xticks([]),plt.yticks([])  
        plt.subplot(224), plt.imshow(img_pre,cmap= "gray")  
        plt.title('Detected Point'),plt.xticks([]),plt.yticks([])  
        plt.show()  
        
    print("success")
