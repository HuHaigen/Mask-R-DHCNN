# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import re
import time
import keras
import numpy as np
import cv2
from matplotlib import pyplot
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
# import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

# 获取当前根目录
ROOT_DIR = os.getcwd()

# 模型保存路径
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# 模型路径
MODEL_PATH = os.path.join(MODEL_DIR, "cell20191221T2029/mask_rcnn_cell_0001.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(MODEL_PATH):
    utils.download_trained_weights(MODEL_PATH)
    print("模型路径为空")


# 定义模型配置，该类继承了 mrcnn.config.Config 类
class CellConfig(Config):
    """用于自定义设置训练参数的配置
    """
    # 给模型对象命名
    NAME = "Cell"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # 设置类的数量（背景+细胞）
    NUM_CLASSES = 1 + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # 每轮训练的迭代数量
    STEPS_PER_EPOCH = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1


config = CellConfig()
config.display()


# 用于定义和加载数据集的类，该类继承了 mrcnn.utils.Dataset类
class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask,为每个像素点绘制0or1的像素值
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # 返回图像中每个位置的像素值
                    at_pixel = image.getpixel((i, j))
                    # 如果当前位置属于目标像素，则将该点的mask数组中的值设置为1
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 自定义的加载函数，负责定义类以及定义数据集中的函数
    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, img_floder, imglist, dataset_root_path):

        # 添加图片中需要检测的类型 (数据集名称，类型编号 从1开始0代表背景，类型名称)
        self.add_class("cell", 1, "cell")  # 细胞
        # 如果有多个分类可以继续添加
        # self.add_class("shapes", 2, "triangle")

        for i in range(count):
            filestr = imglist[i].split(".")[0]
            # 获取的数据集文件路径
            # 图像的mask路径
            mask_path = dataset_root_path + "label_unzip/Frame_" + filestr + "_json/label.png"
            # 图像的标签名路径
            yaml_path = dataset_root_path + "label_unzip/Frame_" + filestr + "_json/info.yaml"
            # 通过cv2解析图像
            img_path = dataset_root_path + "label_unzip/Frame_" + filestr + "_json/img.png"
            cv_img = cv2.imread(img_path)
            # 定义图像对象，将获取到的图像信息作为参数加载到对象当中
            # 不同的数据集格式加载到对象内的参数不同，此处是coco格式的数据集，如果是VOC2007xml类型的，需要解析加载annotation中的信息
            self.add_image("cell", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask，用于给定的image_id加载图像的mask
    # image_id是数据集中每个图像的索引，是add_image方法在添加图像时产生每个图像的添加顺序序列
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        # print("image_id============================", image_id)
        # 图像的信息都存在image_info的字典中，通过image_id获取，字典中的信息是add_image函数添加的有关图像的信息
        info = self.image_info[image_id]
        # number of object
        count = 1
        img = Image.open(info['mask_path'])
        # 获得mask中物体的个数（个数是由标注的类别以及类别中不同个体的数目决定的）
        num_obj = self.get_obj_index(img)
        # 为所有掩膜创建一个数组，每个数组都位于不同的通道
        # 掩膜是一个和图像维度一样的二维数组，数组中不属于对象的位置值为 0，反之则值为 1
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        # 得到重新绘制之后mask数组，其中目标点位置都设置为1
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)  # logical_not逻辑非
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("cell") != -1:
                labels_form.append("cell")
            # elif labels[i].find("triangle") != -1:
            #     # print "column"
            #     labels_form.append("triangle")
        # 通过class_names字典获取类的索引，然后将索引和掩膜一并添加到需要返回的列表class_ids中
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


dataset_root_path = "F:/python/dataset/t24_m/"
# 训练集
train_path = dataset_root_path + "train/"
train_img_floder = train_path + "image"
train_img_list = os.listdir(train_img_floder)
count = len(train_img_list)
# train数据集准备
dataset_train = DrugDataset()  # 实例化训练数据集类
dataset_train.load_shapes(count, train_img_floder, train_img_list, train_path)
dataset_train.prepare()  # 对载入的数据进行规范化
print('Train: %d' % len(dataset_train.image_ids))

# 验证集
val_path = dataset_root_path + "val/"
val_img_floder = val_path + "image"
val_img_list = os.listdir(val_img_floder)
count = len(val_img_list)
# train数据集准备
dataset_val = DrugDataset()  # 实例化训练数据集类
dataset_val.load_shapes(count, val_img_floder, val_img_list, val_path)
dataset_val.prepare()  # 对载入的数据进行规范化
print('Train: %d' % len(dataset_train.image_ids))

# # 测试单张图片是否能正常加载（绘制bounding box 和 mask）
# image_id = 0
# image = dataset_train.load_image(image_id)
# # 绘制图像
# pyplot.imshow(image)
# # 绘制所有掩膜
# mask, class_ids = dataset_train.load_mask(image_id)
# # 从掩膜中提取边框 代码会根据掩膜生成最小外接矩形
# bbox = extract_bboxes(mask)
# # 显示带有掩膜和边框的图像
# display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

# 设定特定的模型用于训练 创建类 mrcnn.model.MaskRCNN 的实例
# mode="training" 指定训练 config 训练参数 model_dir 模型保存路径
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# 加载训练模型权重信息
model.load_weights(MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                                      "mrcnn_bbox", "mrcnn_mask"])
# model.load_weights(model.find_last()[1], by_name=True)
model.keras_model.summary()

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=3,
            layers="all")
