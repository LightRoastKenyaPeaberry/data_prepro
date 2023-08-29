# data_prepro
一些数据预处理的文件。
## 所需环境
opencv-contrib-python==3.4.2.17
opencv-python==3.4.2.17
numpy==1.19.5
python==3.6.13
matplotlib==3.3.4

## 功能
仅针对图像分类数据集的一些处理，包括
+ 分割训练集和验证集
+ 读取灰度/彩色图片
+ 找到数据集中的空白图片
+ 对整个数据集中较少的种类进行5种数据增强（后续会增加）
+ 训练时返回每个epoch的top1和top5

## 数据集的结构
+ root
    + class_1
        + img_1
        + ...
        + img_n
    + ...
    + class_j
        + img_p
        + ...
        + img_q




