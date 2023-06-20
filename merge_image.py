import os
import cv2
import numpy as np
import albumentations as A
import argparse

parser = argparse.ArgumentParser(description="잘린 이미지를 퍼즐처럼 맞추는 기능")

parser.add_argument("input_filname_prefix", type=str, help="이미지 파일이 담긴 경로입니다.")
parser.add_argument("column_num", type=int, help="행의 입력값 입니다.")
parser.add_argument("row_num", type=int, help="열의 입력값 입니다.")
parser.add_argument("output_filename", type=str, help="결과 파일의 이름 입니다.")
args = parser.parse_args()

def HorizontalFlip(img): #각각의 증강 메서드
    augmentations = A.Compose([
        A.HorizontalFlip(p=1),
    ])
    return augmentations(image=img)["image"]

def VerticalFlip(img):
    augmentations = A.Compose([
        A.VerticalFlip(p=1),
    ], p=1)
    return augmentations(image=img)["image"]

def ROTATE_90_CLOCKWISE(img):
    augmentations = A.Compose([
        A.Rotate(limit=(-90, -90), p=1)
    ])
    return augmentations(image=img)["image"]

def ROTATE_90_CLOCK(img):
    augmentations = A.Compose([
        A.Rotate(limit=(90, 90), p=1)
    ])
    return augmentations(image=img)["image"]

def augment(img,i): #이미지가 증강 되었는지 안되었는지 확인하는 경우의 수 메소드
    if i == 0:
        return img
    elif i == 1:
        return HorizontalFlip(img)
    elif i == 2:
        return VerticalFlip(img)
    elif i == 3:
        return ROTATE_90_CLOCKWISE(img)
    elif i == 4:
        return ROTATE_90_CLOCK(img)
    elif i == 5:
        return HorizontalFlip(VerticalFlip(img))
    elif i == 6:
        return HorizontalFlip(ROTATE_90_CLOCKWISE(img))
    elif i == 7:
        return HorizontalFlip(ROTATE_90_CLOCK(img))
    elif i == 8:
        return VerticalFlip(ROTATE_90_CLOCKWISE(img))
    elif i == 9:
        return VerticalFlip(ROTATE_90_CLOCK(img))
    elif i == 10:
        return HorizontalFlip(ROTATE_90_CLOCKWISE(VerticalFlip(img)))
    elif i == 11:
        return HorizontalFlip(ROTATE_90_CLOCK(VerticalFlip(img)))

def merge(img1, img_list, flag): # img1 에 대하여 img2를 모든 경우의 수로 증강시켜보고 score를 매기는 과정.
    lowest = [999999] #[score, img_array, file_name]

    for i in range(len(img_list)):
        low = [999999,99999,0,0,1] #[score,[top,bottom,left,right],augment_num, img_index, img_array]
        
        for j in range(12):
            img2 = augment(img_list[i],j)

            if len(img1[0, :,:]) == len(img2[-1, :,:]):
                top = np.sum((img1[0, :,:] - img2[-1, :,:])**2)
            else:
                top = 99999999
            if len(img1[-1, :,:]) == len(img2[0, :,:]):
                bottom = np.sum((img1[-1, :,:] - img2[0, :,:])**2)
            else:
                bottom = 99999999
            if len(img1[:, 0,:]) == len(img2[:, -1,:]):
                left = np.sum((img1[:, 0,:] - img2[:, -1,:])**2)
            else:
                left = 999999999
            if len(img1[:, -1,:]) == len(img2[:, 0,:]):
                right = np.sum((img1[:, -1,:] - img2[:, 0,:])**2)
            else:
                right = 999999999
            
            if flag == 'ver': # 2:2 비율에서 2가지 이미지가 맞춰졌을 경우 위아래 또는 왼쪽 오른쪽만 경우의 수를 찾는다. (연산량 감소)
                score = [top, bottom]
            elif flag == 'hor':
                score = [left, right]
            elif flag == 'top':
                score = [top]
            elif flag == 'bottom':
                score = [bottom]
            elif flag == 'left':
                score = [left]
            elif flag == 'right':
                score = [right]
            else:
                score = [top, bottom, left, right]

            if min(score) < low[0]:
                low[0] = min(score)
                low[1] = score.index(min(score))
                low[2] = j
                low[3] = i
                low[4] = img2

            if low[0] < lowest[0]: 
                lowest = low

    return lowest

path = args.input_filname_prefix + "/" #image_file_name_prefix
m,x = args.column_num, args.row_num
output = args.output_filename

file_list = os.listdir(path)
img_list = []
result_list = []

for file in file_list: #폴더에 이미지 파일이 없는 경우 오류발생
    if '.DS_Store' == str(file):
        continue
    img = cv2.imread(path+file)
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    img_list.append(img)

flag = ""
if m == 2 and x == 2: # 2:2의 비율로 잘린 이미지의 경우 두이미지만 완성하면 하나의 row가 완성된다.
    merge_img = []
    img1 = np.array(img_list.pop(0),np.uint8)

    for i in range(m):
        
        img2 = merge(img1, img_list, None)

        if img2[1] == 0:
            img = cv2.vconcat([img2[-1],img1])

        elif img2[1] == 1:
            img = cv2.vconcat([img1,img2[-1]])
            
        elif img2[1] == 2:
            img = cv2.hconcat([img2[-1],img1])
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif img2[1] == 3:
            img = cv2.hconcat([img1,img2[-1]])
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        merge_img.append(img)
        if i == m-2:
            img_list.pop(img2[3])

            img1 = img_list.pop(0)
        else:
            pass
    
    for j in range(m-1):
        img1 = merge_img.pop()
        result = merge(img1, merge_img, 'hor')

        if result[1] == 0:
            result = cv2.hconcat([result[-1],img1])

        elif result[1] == 1:
            result = cv2.hconcat([img1,result[-1]])
        
    cv2.imwrite(output+".png",result)

elif m == 3 and x == 3: # 3:3 비율 이미지의 경우 2이미지를 붙여도 row가 완성되지 않아 하나의 이미지를 더 찾아야한다.
    merge_img = []
    temp_list = []
    for j in range(x):
        img1 = np.array(img_list.pop(0),np.uint8)
        img2 = merge(img1, img_list, "ver")
        
        if img2[1] == 0:
            img = cv2.vconcat([img2[-1],img1])
            temp_list = [img2[-1],img1]

        elif img2[1] == 1:
            img = cv2.vconcat([img1,img2[-1]])
            temp_list = [img1,img2[-1]]

        elif img2[1] == 2:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img2[-1] = cv2.rotate(img2[-1], cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.vconcat([img1,img2[-1]])
            temp_list = [img1,img2[-1]]


        elif img2[1] == 3:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img2[-1] = cv2.rotate(img2[-1], cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.vconcat([img1,img2[-1]])
            temp_list = [img2[-1],img1]

            
        img_list.pop(img2[3])

        score = [[9999999]]
        img_flag = True
        
        for k in range(2):# 두이미지를 완성한 후 위쪽 이미지의 top 이미지 또는 아래쪽 이미지의 bottom이미지 중 자연스러운 이미지를 찾는다. (연산량 감소)
            result = None
            if k == 0:
                result = merge(temp_list[k], img_list, 'top')
            elif k == 1:
                result = merge(temp_list[k], img_list, 'bottom')

            if result[0] < score[0][0]:
                score[0] = result
                if k == 0:
                    img_flag = True
                else:
                    img_flag = False

        img_list.pop(score[0][3])
        
        if img_flag:
            if img2[1] == 1 or img2[1] == 2 or img2[1] == 3: 
                result = cv2.vconcat([score[0][-1], img1, img2[-1]])
            else:
                result = cv2.vconcat([score[0][-1], img2[-1], img1])

        else:
            if img2[1] == 1 or img2[1] == 2 or img2[1] == 3:
                result = cv2.vconcat([img1,img2[-1],score[0][-1]])
            else:
                result = cv2.vconcat([img2[-1],img1,score[0][-1]])

        merge_img.append(result)


    img1 = merge_img.pop(0)
    img2 = merge_img.pop(0)
    img3 = merge_img.pop(0)
    
    final_merge = [img2, img3]
    result1 = merge(img1,[img2,img3], 'hor')

    final_merge.pop(result1[3])
    if result1[1] == 1:
        result10 = cv2.hconcat([img1,result1[-1]])
        
        case_left = merge(img1,final_merge, 'left')
        case_right = merge(result1[-1],final_merge, 'right')
        
        if case_left[0] < case_right[0]:
            cv2.imwrite(output+".png",cv2.hconcat([case_left[-1],img1,result1[-1]]))
        else:
            cv2.imwrite(output+".png",cv2.hconcat([img1,result1[-1],case_right[-1]]))
    else:
        result10 = cv2.hconcat([result1[-1],img1])
                       
        case_left = merge(result1[-1],final_merge, 'left')
        case_right = merge(img1,final_merge, 'right')
                       
        if case_left[0] < case_right[0]:
            cv2.imwrite(output+".png",cv2.hconcat([case_left[-1],result1[-1],img1]))
        else:
            cv2.imwrite(output+".png",cv2.hconcat([result1[-1],img1,case_right[-1]]))
