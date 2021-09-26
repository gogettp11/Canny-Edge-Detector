import matplotlib.pyplot as plt 
import numpy as np
import cv2 as cv
import copy

#hyperparameters
HIGH = 80
LOW = 30
MAX_UINT8 = 255

def Canny_detector(img):
    
    img = cv.GaussianBlur(img,(3,3), 1,None,1)

    dx = cv.Sobel(img, ddepth = cv.CV_64F, dx = 1, dy=0,dst = None,ksize=3)
    dy = cv.Sobel(img, ddepth = cv.CV_64F, dx = 0, dy=1,dst = None,ksize=3)

    gradient_magnitude = cv.sqrt(cv.pow(dx,2)+cv.pow(dy,2))
    angles = np.arctan2(dx,dy) * 180/np.pi
    angles = np.abs(angles)


    temp_var = copy.deepcopy(img)
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            q = MAX_UINT8
            r = MAX_UINT8
            #angle 0
            if (0 <= angles[i,j] < 22.5) or (157.5 <= angles[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
                #angle 45
            elif (22.5 <= angles[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
                #angle 90
            elif (67.5 <= angles[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
                #angle 135
            elif (112.5 <= angles[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]
            
            if q < gradient_magnitude[i,j] and r < gradient_magnitude[i,j]:
                temp_var[i,j] = gradient_magnitude[i,j]
            else:
                temp_var[i,j] = 0

    temp_var_2 = copy.deepcopy(temp_var)
    #histeresis thresholds: high > 200 , low > 120
    for i in range(1,len(img)-1):
        for j in range(1,len(img[i])-1):
            if temp_var[i,j] > HIGH:
                temp_var_2[i,j] = MAX_UINT8
            elif temp_var[i,j] > LOW and check_neigbours_for_high_value(temp_var,i,j):
                temp_var_2[i,j] = MAX_UINT8
            else:
                temp_var_2[i,j] = 0

    return temp_var_2

def check_neigbours_for_high_value(gradient_img, i,j):
    if gradient_img[i-1,j-1] > HIGH or gradient_img[i+1,j+1] > HIGH or \
        gradient_img[i-1,j+1] > HIGH or gradient_img[i+1,j-1] > HIGH or \
            gradient_img[i,j-1] > HIGH or gradient_img[i,j+1] > HIGH or\
                gradient_img[i+1,j] > HIGH or gradient_img[i-1,j] > HIGH:
                    return True

    return False

def visualize(imgs, format=None):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(3, 3, plt_idx)    
        plt.imshow(img, cmap=format)
    plt.show()

def load_data(dir_name = './plates'):
    return [cv.imread(f"{dir_name}/1.png",cv.IMREAD_GRAYSCALE),cv.imread(f"{dir_name}/2.png",cv.IMREAD_GRAYSCALE),cv.imread(f"{dir_name}/3.png",cv.IMREAD_GRAYSCALE)]

plates = load_data()

canny_imgs = []
for i,img in enumerate(plates):
    canny_img = Canny_detector(img)
    cv.imwrite(f"./borders/{i}.jpg",canny_img)
    canny_imgs.append(canny_img)

visualize(canny_imgs, 'gray')