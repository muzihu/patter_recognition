import cv2  
import numpy as np  
import matplotlib
import matplotlib.pyplot as plt

def haar_extractor(img_1, img_2):
    """
    输入一个图片，计算该图的harr特征
    img_1 :白色图
    img_2 ：阴影图
    :return: 白色图与阴影图之差的loss
    size same
    """
    num1 = sum(sum(img_1))
    num2 = sum(sum(img_2))
    return num1 - num2

def cut_picture(img, x, y, w, h):
    """
    函数功能：
    输入一个图片，以及切割参数，返回切割好后的二值化图片
    参数：
    img：图片输入
    x, y: 希望切割的图片的左上角坐标
    w, h: 图片的宽度和高度
    flag: flag = 1 表示切下的图片是外接的
    """
    cut_img = img[x:x+h, y:y+w]
    mean = cut_img.mean()
    cut_img[cut_img <= mean] = 0
    cut_img[cut_img > mean] = 255
    return np.array(cut_img)

def feature_extractor(img):
    """
    输入一个图片，抽取该图片的特征向量
    img: 输入图像 尺寸大小为30 × 30 
    feature: 输出的特征向量 
    32*32  ——》   31*28
    """
    feature = []
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    # 以2×2为一个方块抽取哈尔特征：
    for i in range(0, 32, 4):
        for j in range(0, 32, 2):
            split_img = cut_picture(img, i, j, 2, 4)
            split_img[split_img == 255] = 1
            num = haar_extractor(np.array(split_img[:2,:]), np.array(split_img[2:,:]))
            feature.append(num)
    # 以2×4为一个方块提取哈尔特征：
    for i in range(0, 32, 4):
        for j in range(0, 32, 4):
            split_img = cut_picture(img, i, j, 4, 4)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:2,:], split_img[2:,:])
            feature.append(num)
    # 以4×4为一个方块提取哈尔特征：
    for i in range(0, 32, 8):
        for j in range(0, 32, 4):
            split_img = cut_picture(img, i, j, 4, 8)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:4,:], split_img[4:,:])
            feature.append(num)
    # 以8×16为一个方块提取哈尔特征：
    for i in range(0, 32, 16):
        for j in range(0, 32, 16):
            split_img = cut_picture(img, i, j, 16, 16)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:8, :], split_img[8:, :])
            feature.append(num)
    # 以16×16为一个方块提取哈尔特征：
    for i in range(0, 32, 32):
        for j in range(0, 32, 16):
            split_img = cut_picture(img, i, j, 16, 32)
            split_img[split_img == 255] = 1
            num = haar_extractor(split_img[:16, :], split_img[16:, :])
            feature.append(num)
    return feature

# def sample_haar_cal(img, size):
#     sample_haar = np.zeros(range(len(img)-int(size[0])),range(len(img[0])-int(size[1])))
#     for i in range(len(img)-int(size[0])):
#         for j in range(len(img[0])-int(size[1])):
#             img_tmp = img[i:i+int(size[0]),j:j+int(size[1])]
#             sample_haar[i][j] = feature_extractor(img_tmp)
#     return sample_haar

def matchTemplate(img,mod_haar,size):
    # sample_haar = sample_haar_cal(img, size)
    # for i in range(len(mod_haar)):
    #     for j in range(len(img[0])-int(size[1])):
    #         img_tmp = img[i:i+int(size[0]),j:j+int(size[1])]
    #         sample_haar[i][j] = feature_extractor(img_tmp)
    #         loss = sample_haar - mod_haar[i]
    #         loss = sum(loss ** 2)
    # sample_haar = np.zeros((len(img)-int(size[0]),len(img[0])-int(size[1])))
    res = np.zeros((len(img)-int(size[0]),len(img[0])-int(size[1])))
    for i in range(len(res)):
        for j in range(len(res[0])):
            img_tmp = img[i:i+int(size[0]),j:j+int(size[1])]
            sample_haar = feature_extractor(img_tmp)
            res[i][j] = sum((np.array(sample_haar) - np.array(mod_haar)) ** 2)
    return res

if __name__ == '__main__':
    test_num = ["1","2","3","4","5","6","划痕","噪声"]  
    for test_test in range(8): 
        img_pre = cv2.imread("./test/"+test_num[test_test]+".bmp",0)  
        img=cv2.adaptiveThreshold(img_pre,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,35,5)
        # ADAPTIVE_THRESH_GAUSSIAN_C
        img2 = img.copy()  
        template_num = ["0","1","2","3","4","6","8","9"]
        w = np.zeros((8,1))
        h = np.zeros((8,1))

        img = img2.copy()  
        meth = 'cv2.TM_CCOEFF'
        method = eval(meth)  
        template_pre = {}
        mod_haar = {}
        res = {}

        for i in range(8):
            template_pre_single = cv2.imread("./train/"+template_num[i]+".bmp",0)  
            ret_single,template_single = cv2.threshold(template_pre_single,template_pre_single.mean(),255,cv2.THRESH_BINARY)
            w[i],h[i] = template_single.shape[::-1]  
            mod_haar[i] = feature_extractor(template_pre_single)
            template_pre[i] = template_single

        for i in range(8): 
            tmp1 = int((max(w)-w[i])/2)
            tmp2 = int(max(w)-w[i]-tmp1)
            tmp3 = int((max(h)-h[i])/2)
            tmp4 = int(max(h)-h[i]-tmp3)
            template_pre[i] = cv2.copyMakeBorder(template_pre[i], tmp3,tmp4,tmp1,tmp2, cv2.BORDER_CONSTANT, value=[255,255,255])
            template_pre[i] = cv2.resize(template_pre[i], (32, 32), interpolation=cv2.INTER_CUBIC)
            # 步长为1的扫描窗所得残差res 
            size = [max(h), max(w)]
            res_tmp = matchTemplate(img,mod_haar[i],size) 
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

        if test_test<6:
            threshold_i = 7e6
        elif test_test==6:
            threshold_i = 6e6
        elif test_test==7:
            threshold_i = 4.3e6
        else:
            break
        cho_max = 14
        top_left = {}
        cho_num = {}
        # i = 0
        for i in range(cho_max):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_max)  
            # if max_val<threshold_i:
            #     break
            top_left[i] = max_loc  
            bottom_right = (max_loc[0] + int(max(w)), max_loc[1] + int(max(h)))
            # 通过确定对角线画矩形
            cv2.rectangle(img_pre,max_loc, bottom_right, 255, 2) 
            cho_num[i] = num_like[max_loc[1]][max_loc[0]] 
            img_save = img_pre[max_loc[1]:max_loc[1]+int(max(h)),max_loc[0]:max_loc[0]+int(max(w))]
            for j in range(max(w)/2):
                for k in range(max(h)/2):
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
            # cv2.imwrite('./'+test_num[test_test]+'/'+str(i)+'_'+template_num[int(cho_num[i])]+'.jpg',img_save)
            # i += 1
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
