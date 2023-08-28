'''
暂定对少于50张的字， 做增强。
请注意数据的结构。给定的start_path包含的是类别文件夹，各个类别文件夹里才是图片。
'''
import numpy as np
import cv2 as cv
import os

# import count_files


def dilate(source: str, dest: str):
    '''
    Args:
        source: the abspath of the img
        dest: the abspath of the dir to place the result
    '''
    src = cv.imread(source, cv.IMREAD_UNCHANGED)

    #设置卷积核
    kernel = np.ones((5,5), np.uint8)

    #图像膨胀处理
    dilation = cv.dilate(src, kernel)

    pref = source.split('.')[0]

    try:
        cv.imwrite(os.path.join(dest, f'{pref}_dil.jpg'), dilation)
    except:
        print('保存失败')
    print('成功')
    

def erode(source, dest):
    src = cv.imread(source, cv.IMREAD_UNCHANGED)

    #设置卷积核
    kernel = np.ones((5,5), np.uint8)

    #图像膨胀处理
    erosion = cv.erode(src, kernel)

    pref = source.split('.')[0]

    try:
        cv.imwrite(os.path.join(dest, f'{pref}_ero.jpg'), erosion)
    except:
        print('保存失败')
    print('成功')



def open(source, dest):
    # 开：先腐蚀，再膨胀
    img = cv.imread(source, cv.IMREAD_UNCHANGED)

    kernel = np.ones((5,5),np.uint8) 
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    pref = source.split('.')[0]

    try:
        cv.imwrite(os.path.join(dest, f'{pref}_open.jpg'), opening)
    except:
        print('保存失败')
    print('成功')   


def close(source, dest):
    # 开：先腐蚀，再膨胀
    img = cv.imread(source, cv.IMREAD_UNCHANGED)

    kernel = np.ones((5,5),np.uint8) 
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    pref = source.split('.')[0]

    try:
        cv.imwrite(os.path.join(dest, f'{pref}_close.jpg'), closing)
    except:
        print('保存失败')
    print('成功')


def blur(source, dest):

    img = cv.imread(source, cv.IMREAD_UNCHANGED)

    blured = cv.medianBlur(img,5)
    pref = source.split('.')[0]

    try:
        cv.imwrite(os.path.join(dest, f'{pref}_blured.jpg'), blured)
    except:
        print('保存失败')
    print('成功')


def count_files(start_path):
    '''
    仅适用于根目录(start_path)下有很多子目录， 子目录里都是文件的形式， 其他情况请酌情改动。== 三层的树
    '''
    subdirs = os.listdir(start_path)
    file_num_per_dir = []
    for i in range(len(subdirs)):
        file_num_per_dir.append(len(os.listdir(os.path.join(start_path, subdirs[i]))))

    return file_num_per_dir


def find_small_dir(start_path: str, threshhold: int):
    '''
    仅适用于根目录(start_path)下有很多子目录， 子目录里都是文件的形式， 找到文件数量小于threshhold的子目录。
    '''
    subdirs = os.listdir(start_path)
    file_num_per_dir = []
    for i in range(len(subdirs)):
        file_num_per_dir.append(len(os.listdir(os.path.join(start_path, subdirs[i]))))

    file_num_per_dir = np.array(file_num_per_dir)
    small_dir_idx = np.where(file_num_per_dir < threshhold)[0]
    small_dir = []
    for i in small_dir_idx:
        small_dir.append(subdirs[i])

    return small_dir


def five_augment(start_path: str, threshhold: int):
    '''建议将数据集存为两份， 一份不做改动， 另一份用来做数据增强'''
    small_dir = find_small_dir(start_path, threshhold)
    for dir in small_dir:
        dir_path = os.path.join(start_path, dir)
        imgs = os.listdir(dir_path)
        for img in imgs:
            img_path = os.path.join(dir_path, img)
            img_path = os.path.normpath(img_path)
            # 注意img_path 里不能包括中文，不然会报错
            dilate(img_path, dir_path)
            erode(img_path, dir_path)
            open(img_path, dir_path)
            close(img_path, dir_path)
            blur(img_path, dir_path)


def rename_dir(start_path):
    '''这里重命名是按照现有的文件夹改的，至于其他特点的文件夹依情况再修改。'''
    subdirs = os.listdir(start_path)
    for dir_name in subdirs:
        os.rename(os.path.join(start_path, dir_name), os.path.join(start_path, dir_name[:4]))


if __name__ == '__main__':
    rename_dir('/home/zxy/imgs/CharSample')
    five_augment('/home/zxy/imgs/CharSample', 50)