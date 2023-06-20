import os
import cv2
import numpy as np
import random
import albumentations as A
import argparse
import uuid

parser = argparse.ArgumentParser(description="이미지를 자르고 증강하는 기능")

parser.add_argument("image_file_name", type=str, help="이미지 파일의 이름 입니다.")
parser.add_argument("column_num", type=int, help="행의 입력값 입니다.")
parser.add_argument("row_num", type=int, help="열의 입력값 입니다.")
parser.add_argument("prefix_output_filename", type=str, help="결과 파일의 경로 설정 입니다.")

args = parser.parse_args()

def slice_img(img,m,x):
    row, col, _ = img.shape
    if col%m != 0:
        img = img[:, col%m:, :]
    if row%x != 0:
        img = img[row%x:, :, :]
    row, col, _ = img.shape
    
    width =  int(col // m)
    height =  int(row // x)

    aug_img_list = []
    for i in range(0, col, width):
        for j in range(0, row, height):
            aug_img = augment(img[j:j + height, i:i + width])
            aug_img_list.append(aug_img)
            
    os.mkdir(args.prefix_output_filename)
    random.shuffle(aug_img_list)
    for n in aug_img_list:
        cv2.imwrite(str(args.prefix_output_filename)+"/"+str(uuid.uuid4())+".png",n)
        print(n.shape)
    
def rotate_right(image, **kwargs):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def augment(img):
    augmentations = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Lambda(image=rotate_right, p=0.5)
    ], p=1)
    augmented = augmentations(image=img)
    augments= augmented["image"]
    return augments

m, x = args.column_num, args.row_num

img = cv2.imread(args.image_file_name)

col, row = img.shape[0], img.shape[1]

img = slice_img(img,m,x)