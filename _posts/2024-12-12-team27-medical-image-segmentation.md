---
layout: post
comments: true
title: Medical Image Segmentation
author: Om Patel, Suyeon Shin, Harkanwar Singh, Emmett Cocke
date: 2024-12-12
---


> This report covers medical image segmentation using U-Net, U-NET++, and PSPNet. These models are ran on ISIC and AMOS datasets. 


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
Medical image segmentation involves taking a medical image obtained from MRI/CT/etc. and segmenting different parts of the image. More specifically, given a 2D or 3D medical image, the goal is to produce a segmentation mask of the same dimension as the input. Labels in this mask correspond semantically to relevant parts of the image that are classified based on predefined classes. Some examples of classes include background, organ, and lesion. Unlike normal image classification tasks or most image segmentation tasks, medical image segmentation requires a very high level of accuracy. Radiologists who work with these images are trained for years to be able to accurately segment these images and identify lesions accurately. However, there are many more medical images than doctors that can make accurate diagnoses in the world, and so medical image segmentation attempts to solve this problem. Essentially, medical image segmentation aims to assist radiologists in detecting lesions from medical image scans and identifying regions of interest in an organ, reducing the amount of time radiologists may take to annotate these images. The models we will explore in this paper to implement this solution are U-Net, U-Net++, and PSPNet. 

## Dataset
There are two datasets we will use in this model. One comes from ISIC (International Skin Imaging Collaboration).  
## Model 1: U-Net
### Architecture
### Training
### Results
### Discussion



## Model 2: U-Net++
### Architecture
### Training
### Results
### Discussion



## Model 3: TBD
### Architecture
### Training
### Results
### Discussion


## Conclusion


## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
