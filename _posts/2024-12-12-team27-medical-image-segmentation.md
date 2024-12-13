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
The dataset used for this paper comes from ISIC (International Skin Imaging Collaboration), specifically the ISIC challenge dataset from 2017 [1]. The training dataset consists of 2000 skin lesion images in JPEG format and 2000 superpixel masks in PNG format. The ground truth training data consists of 2000 binary mask images in PNG format, 2000 dermoscopic feature files in JSON format, and 2000 lesion diagnoses. The goal of this dataset was train models to accurately diagnose melanoma. The challenge using this dataset involved segmenting images, feature detection, and disease classification. For the purposes of this paper, we will only focus on segmenting medical images, meaning we will not be using the 2000 dermoscopic feature files and 2000 lesion diagnoses. 

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



## Model 3: PSPNet
### Architecture
### Training
### Results
### Discussion


## Conclusion


## Reference

[1]  Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A, Liopyris K, Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)". arXiv: 1710.05006 

---
