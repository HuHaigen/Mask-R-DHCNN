# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import yaml
import cv2
from PIL import Image
from mrcnn.config import Config

# 获取当前根目录
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# 模型保存路径
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# 模型路径
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_0040.h5")


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
    IMAGE_MIN_DIM = 1600
    IMAGE_MAX_DIM = 1600

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # 每轮训练的迭代数量
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


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
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):

        # 添加图片中需要检测的类型 (数据集名称，类型编号 从1开始0代表背景，类型名称)
        self.add_class("cell", 1, "cell")  # 细胞
        # 如果有多个分类可以继续添加
        # self.add_class("shapes", 2, "triangle")

        for i in range(count):
            filestr = imglist[i].split(".")[0]
            # 获取的数据集文件路径
            # 图像的mask路径
            mask_path = mask_floder + "/label" + filestr + ".png"
            # 图像的标签名路径
            yaml_path = dataset_root_path + "label_unzip/Frame_" + filestr + "_json/info.yaml"
            print(dataset_root_path + "label_unzip/Frame_" + filestr + "_json/img.png")
            # 通过cv2解析图像
            cv_img = cv2.imread(dataset_root_path + "label_unzip/Frame_" + filestr + "_json/img.png")
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


# 测试集路径设置
dataset_root_path = "images/t24/t24_label/train/"
img_floder = dataset_root_path + "image"
mask_floder = dataset_root_path + "label"
imglist = os.listdir(img_floder)
count = len(imglist)

dataset_val = DrugDataset()
dataset_val.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()


# 简化训练模型配置
class InferenceConfig(CellConfig):
    # 简化 GPU 配置
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

# 设定特定的模型用于测试 创建类 mrcnn.model.MaskRCNN 的实例
# mode="inference" 指定测试 config 测试参数 model_dir 模型保存路径
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# 加载权重
model.load_weights(MODEL_PATH, by_name=True)

class_names = ['BG', 'cell']


# 计算单张分割精度
# local 为预测目标mask,ground_truth 为实际目标mask
def compute_AP(local, ground_truth):
    overlap_area = 0
    mask_area = 0
    FP = 0
    FN = 0
    for i in range(1600):
        for j in range(1600):
            if ground_truth[i][j].any():
                mask_area += 1
            for k in range(local.shape[2]):
                if (local[i][j][k] == ground_truth[i][j]).any() and ground_truth[i][j].any():
                    overlap_area += 1
                if local[i][j][k].any() and (ground_truth[i][j] != local[i][j][k]).any():
                    FP += 1
                if (local[i][j][k] != ground_truth[i][j]).any() and ground_truth[i][j].any():
                    FN += 1
    print("overlap_area", overlap_area)
    print("mask_area:", mask_area)
    TP = overlap_area
    P = TP / (TP + FP)
    return P


# 计算给定数据集中模型的 mAP 并将结果打印出来
def evaluate_model(dataset_val, config, model):
    myAPs = []
    APs = []
    for image_id in dataset_val.image_ids:
        # 根据指定的 image_id 从数据集中加载出图像和真实掩膜
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, config,
                                                                                  image_id, use_mini_mask=False)
        # 转换像素值并且在图像上扩展一个维度作为模型预测的输入
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # 进行检测
        results = model.detect(molded_images, verbose=0)
        # 获取结果
        r = results[0]
        # 将测试结果打印出来
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
        # 计算AP
        # Compute VOC-Style mAP @ IoU=0.5
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        print("AP", AP)
        print("precisions: ", precisions)
        print("recalls: ", recalls)
        print("overlaps: ", overlaps)
        myAPs.append(compute_AP(r['masks'], gt_mask))
        print(compute_AP(r['masks'], gt_mask))


evaluate_model(dataset_val, config, model)
