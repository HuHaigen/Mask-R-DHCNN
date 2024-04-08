# [Dilated Heterogeneous Convolution for Cell Detection and Segmentation Based on Mask R-CNN](https://github.com/HuHaigen/Mask-R-DHCNN)

> **The Paper Links:** [To be supplemented]()
> 
> **Authors:** [Fengdan Hu, ](),  [Haigen Hu, ](),  [Hui Xu, ](),  [Jinshan XU](),  *[Qi Chen]()*, 

## Abstract

Owing to the variable shapes, large size difference, uneven grayscale, and dense distribution among biological cells in an image, it is very difficult to accurately detect and segment cells. Especially, it is a serious challenge for some microscope imaging devices with limited resources due to a large number of learning parameters and computational burden when using the standard Mask R-CNN. In this work, we proposed a mask R-DHCNN for cell detection and segmentation. More specifically, Dilation Heterogeneous Convolution (DHConv) is proposed by designing a novel convolutional kernel structure (i.e., DHConv), which integrates the strengths of the heterogeneous kernel structure and dilated convolution. Then the traditional homogeneous convolution structure of the standard Mask R-CNN is replaced with the proposed DHConv module to adapt to shapes and sizes differences encountered in cell detection and segmentation tasks. Finally, a series of comparison and ablation experiments are conducted on various biological cell datasets (such as U373, GoTW1, SIM+, and T24) to verify the effectiveness of the proposed method. The results show that the proposed method can obtain better performance than some state-of-the-art methods in multiple metrics (including AP, Precision, Recall, Dice, and PQ) while maintaining competitive FLOPs and FPS.

## 1. Introduction

<p align="center">
  <img src="https://github.com/HuHaigen/Mask-R-DHCNN/blob/main/figs/DHConv.png"></img>
	<em>
	Fig. 1. Dilated Heterogeneous Convolution (DHConv) module.
	</em>
</p>


## 2. Experiments

We have done a series of qualitative and quantitative experimental comparisons on our proposed method, please refer to the paper ([Link]()) for the specific experimental results. The following is a brief data description and experimental results.

- U373 Dataset

| Methods            | AP (%)     | Precision (%) | Recall (%) | Dice (%)   | FPS   |
| ------------------ | ---------- | ------------- | ---------- | ---------- | ----- |
| Baseline           | 91.39±0.33 | 87.21±0.30    | 79.31±0.27 | 82.03±0.47 | 4.81  |
| MS R-CNN           | 90.12±0.12 | 86.35±0.17    | 78.90±0.25 | 81.87±0.31 | 4.81  |
| ExtremeNet         | 77.75±1.11 | 71.68±1.27    | 60.55±0.89 | 70.11±0.77 | 4.73  |
| TensorMask         | 83.37±1.51 | 79.92±1.93    | 68.31±2.11 | 78.41±0.88 | 2.47  |
| PolarMask          | 88.77±0.10 | 83.09±0.07    | 71.93±0.21 | 80.85±0.13 | 11.79 |
| CenterMask         | 79.33±1.74 | 72.40±1.82    | 61.40±1.19 | 74.90±1.85 | 7.15  |
| ResNet-50-FPN-ISO  | 92.74±0.47 | 88.65±0.19    | 83.04±0.27 | 82.81±0.31 | -     |
| Mask R-DHCNN(Ours) | 92.87±0.53 | 88.26±0.06    | 80.52±0.18 | 84.21±0.80 | 7.00  |

- GoTW1 Dataset

| Methods            | AP (%)     | Precision (%) | Recall (%) | Dice (%)   | FPS  |
| ------------------ | ---------- | ------------- | ---------- | ---------- | ---- |
| Baseline           | 90.64±0.44 | 91.14±0.48    | 87.66±0.23 | 89.65±0.33 | 4.00 |
| MS R-CNN           | 88.77±0.64 | 89.26±0.88    | 85.38±0.72 | 86.05±0.59 | 3.95 |
| ExtremeNet         | 84.40±1.03 | 86.75±1.21    | 80.51±0.89 | 82.37±1.22 | 3.90 |
| TensorMask         | 80.09±0.97 | 76.27±1.17    | 70.44±0.82 | 76.27±1.19 | 2.00 |
| PolarMask          | 85.65±0.83 | 87.00±0.77    | 83.43±0.50 | 85.98±0.81 | 9.50 |
| CenterMask         | 78.10±1.56 | 74.51±2.10    | 67.39±1.68 | 73.71±1.40 | 6.13 |
| ResNet-50-FPN-ISO  | 91.18±1.07 | 92.26±0.89    | 90.99±1.14 | 91.05±0.59 | -    |
| Mask R-DHCNN(Ours) | 91.26±0.85 | 91.84±1.23    | 88.99±0.94 | 90.61±0.65 | 6.70 |

- SIM+01 Dataset

| Methods            | AP (%)     | Precision (%) | Recall (%) | Dice (%)   | FPS   |
| ------------------ | ---------- | ------------- | ---------- | ---------- | ----- |
| Baseline           | 93.93±0.69 | 94.06±0.21    | 86.18±0.58 | 87.60±0.40 | 4.10  |
| MS R-CNN           | 92.03±0.27 | 93.10±0.89    | 85.86±0.33 | 86.38±0.64 | 4.00  |
| ExtremeNet         | 88.64±1.45 | 90.49±1.29    | 81.30±1.37 | 83.24±1.01 | 3.84  |
| TensorMask         | 87.24±0.93 | 89.94±1.39    | 80.80±1.27 | 83.05±1.71 | 2.15  |
| PolarMask          | 91.19±1.13 | 92.08±0.97    | 84.65±0.48 | 85.74±0.70 | 10.05 |
| CenterMask         | 85.31±1.66 | 88.38±1.02    | 80.29±1.85 | 78.77±1.90 | 6.20  |
| ResNet-50-FPN-ISO  | 94.87±0.44 | 94.79±0.39    | 84.67±0.61 | 89.66±0.57 | -     |
| Mask R-DHCNN(Ours) | 94.04±1.23 | 94.36±0.87    | 88.03±0.42 | 90.13±0.54 | 5.50  |

- SIM+02 Dataset

| Methods            | AP (%)     | Precision (%) | Recall (%) | Dice (%)   | FPS  |
| ------------------ | ---------- | ------------- | ---------- | ---------- | ---- |
| Baseline           | 80.88±1.05 | 83.95±1.06    | 80.69±1.88 | 75.71±1.24 | 3.81 |
| MS R-CNN           | 88.43±1.07 | 87.92±1.21    | 85.49±1.53 | 83.10±1.56 | 3.75 |
| ExtremeNet         | 73.22±2.71 | 72.49±1.88    | 70.20±2.47 | 70.17±1.06 | 3.50 |
| TensorMask         | 75.41±0.91 | 74.18±0.54    | 70.77±1.23 | 71.20±1.51 | 2.30 |
| PolarMask          | 78.52±0.99 | 79.06±1.15    | 74.36±1.22 | 74.18±0.85 | 9.23 |
| CenterMask         | 70.63±2.92 | 69.30±3.05    | 67.27±1.87 | 66.98±1.28 | 5.75 |
| ResNet-50-FPN-ISO  | 84.06±0.76 | 85.78±1.02    | 83.37±1.00 | 75.64±0.77 | -    |
| Mask R-DHCNN(Ours) | 82.47±1.11 | 85.71±0.79    | 80.24±2.13 | 78.07±0.94 | 5.61 |

- T24 Dataset

| Methods            | AP (%)     | Precision (%) | Recall (%) | Dice (%)   | FPS   |
| ------------------ | ---------- | ------------- | ---------- | ---------- | ----- |
| Baseline           | 92.25±0.83 | 88.25±0.76    | 85.18±0.73 | 93.81±0.56 | 4.28  |
| MS R-CNN           | 91.98±0.07 | 87.67±0.11    | 83.41±0.29 | 93.53±0.31 | 4.29  |
| ExtremeNet         | 81.86±0.88 | 80.88±0.76    | 71.54±0.34 | 79.66±0.50 | 4.12  |
| TensorMask         | 87.53±1.20 | 83.24±1.09    | 76.54±1.52 | 86.33±0.64 | 2.19  |
| PolarMask          | 91.67±0.19 | 86.08±0.20    | 83.10±0.44 | 92.79±0.37 | 10.32 |
| CenterMask         | 82.01±1.14 | 84.80±0.95    | 73.98±1.08 | 80.89±0.74 | 6.42  |
| ResNet-50-FPN-ISO  | 93.41±0.66 | 92.14±0.61    | 83.67±0.71 | 93.82±0.33 | -     |
| Mask R-DHCNN(Ours) | 94.32±0.85 | 91.38±0.56    | 87.15±0.98 | 94.31±0.54 | 6.44  |
