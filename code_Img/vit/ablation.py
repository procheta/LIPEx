#!/usr/bin/env python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

import torch
from torchvision import models, transforms
import torch.nn.functional as F

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_image import LimeImageExplainer

import random
# random.seed(42)
# import time
# strat_time = time.time()

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            print(img.mode)
            return img.convert('RGB')
 
#For Pytorch, first we need to define two separate transforms: 
# (1) to take PIL image, resize and crop it 
# (2) take resized, cropped image and apply weights.
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224)
    ])    
    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    
    return transf    

#  load fine-tuned model
model = models.vit_b_16(weights='DEFAULT')
model.heads[0] = torch.nn.Linear(768, 10)
model.aux_logits = False
model.load_state_dict(torch.load('/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Img/vit/model-3.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))
  

pil_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

#Now we are ready to define classification function that Lime needs. 
#The input to this function is numpy array of images where each image is ndarray of shape **(channel, height, width)**. 
#The output is numpy array of shape (image index, classes) where each value in array should be probability for that image, class combination.

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    model.to(device)
    batch = batch.to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

img_directory ='/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/Dataset/ImageNet/imagenette-320px/test' # 3925 

def RandomImages():
    image_list = os.listdir(img_directory)
    random.shuffle(image_list)
    print('Number of available images:', len(image_list),'\n')
    return image_list

# Prepare segmentation model
sam_checkpoint =  "/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/segment-anything/checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

def segment_image(image=None,image_seg_path=None):
    # image: (H, W, 3) numpy array
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=(lambda x: x['area'])) # sort by pixel area ascending
    LIPEx_segments = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for idx in range(len(masks))[::-1]:
        cur_mask = masks[idx]['segmentation']
        LIPEx_segments[cur_mask] = idx
    # print('Segmentation done by SA')
    return LIPEx_segments

def rePredict(image,TopK_segments,segments,number_of_segments_to_remove):
    count_list = []
    for num in number_of_segments_to_remove:
        count_change = 0
        copy_image = np.copy(image)
        for i in range(num):
            removed_segment = TopK_segments[i]
            copy_image[segments == removed_segment] = 0
        re_pred = batch_predict([copy_image])
        re_pred_class = re_pred.squeeze().argmax()
        if re_pred_class != pred_class:
            count_change = 1
        count_list.append(count_change)
    return count_list


LIME_count, LIPEx_count = [], []

TopK = 5
number_of_segments_to_remove = [1,2,3,4,5]
num_samples = 1000
num_classes = 10

batch_size = 128 #128 for 3090, 512 for A100
union_num = 3  # feature union number
num_of_test_image = 10

for image_name in RandomImages():
    
    print('Image:',image_name)
    # load the image
    image_path = os.path.join(img_directory, image_name)
    image_RGB = get_image(image_path)
    pil_transf_image = pil_transf(image_RGB)
    image = np.array(pil_transf_image)
    # Test predictor for the imput image.
    pred = batch_predict([pil_transf_image])

    pred_class = pred.squeeze().argmax()
    print('Top predict class:',pred_class)

    '''********************* Start LIPEx  *****************************************'''

    LIPEx_explainer = LimeImageExplainer(random_state=42)

    LIPEx_exp = LIPEx_explainer.explain_instance_new(image, 
                                                    batch_predict,
                                                    hide_color=0,
                                                    num_features=union_num,
                                                    num_samples=num_samples,
                                                    top_labels = num_classes,
                                                    batch_size = batch_size, 
                                                    segmentation_fn=segment_image
                                                    )
    after_union = len(LIPEx_exp.used_features)
    print('LIPEx_exp.used_features:', LIPEx_exp.used_features)

    # change the type of local_exp from numpy.ndarray to dict
    pred_weights = LIPEx_exp.local_exp[pred_class]
    pred_sorted_indices = np.argsort([np.abs(w) for w in pred_weights])[::-1]
    weights_with_feature_index = []
    for arr in LIPEx_exp.local_exp:
        weights_with_feature_index.append([(LIPEx_exp.used_features[i], w) for i, w in enumerate(arr)])
    l_exp = {}
    for idx, row in enumerate(weights_with_feature_index):
        new_row = []
        for jdx in pred_sorted_indices:
            new_row.append(row[jdx])
        l_exp[idx] = new_row
    LIPEx_exp.local_exp = l_exp

    segments = LIPEx_exp.segments
    # LIPEx
    LIPEx_ranked_segs = [x[0] for x in LIPEx_exp.local_exp[pred_class]][:TopK]
    print('LIPEx_TopK_segments:',LIPEx_ranked_segs)
    if len(LIPEx_ranked_segs) < TopK:
        print ('LIPEx_ranked_segs is less than',TopK)
        continue
    # LIPEx Re-Prediction 
    LIPEx_count_list = rePredict(image,LIPEx_ranked_segs,segments,number_of_segments_to_remove)
    print('LIPEx_count_list:',LIPEx_count_list)
    
    '''*************************End LIPEx *************************************'''  


    '''**************************** Start LIME *******************************'''
    LIME_explainer = LimeImageExplainer(random_state=42)
    LIME_exp = LIME_explainer.explain_instance(image, 
                                                batch_predict, 
                                                hide_color=0,
                                                top_labels=1,
                                                num_features=after_union,
                                                num_samples=num_samples,
                                                batch_size=batch_size,
                                                segmentation_fn=segment_image,
                                                ) 
    
    used_features = LIME_exp.used_features
    print('LIME used_features:', used_features)
    # LIME
    LIME_ranked_segs = [x[0] for x in LIME_exp.local_exp[pred_class]][:TopK]
    print('LIME_TopK_segments:',LIME_ranked_segs)
    if len(LIME_ranked_segs) < TopK:
        print ('LIME_ranked_segs is less than',TopK)
        continue

    # LIME Re-Prediction
    LIME_count_list = rePredict(image,LIME_ranked_segs,segments,number_of_segments_to_remove)
    print('LIME_count_list:',LIME_count_list)

    '''*************************End LIME *************************************'''  

    LIPEx_count.append(LIPEx_count_list)
    LIME_count.append(LIME_count_list)

    assert len(LIPEx_count) == len(LIME_count)
    if len(LIPEx_count) == num_of_test_image:
        break

LIME_array = np.array(LIME_count)
LIPEx_array = np.array(LIPEx_count)

average_LIME = LIME_array.sum(axis=0) / num_of_test_image
average_LIPEx = LIPEx_array.sum(axis=0) / num_of_test_image
print('LIME:',average_LIME)
print('LIPEx:',average_LIPEx)                     
