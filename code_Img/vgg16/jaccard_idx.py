#!/usr/bin/env python
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_image import LimeImageExplainer

import math
import random
random.seed(42)
import time
start = time.time()

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
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
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

# load the pre-trained model
# model = models.vgg16(weights='DEFAULT')

# load fine-tuned model
model = models.vgg16()
model.classifier[6] = torch.nn.Linear(4096, 10)
model.aux_logits = False
model.load_state_dict(torch.load('/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Img/vgg16/best_ckpt/model-2.pt'))
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

def RandomImages(num_of_test_image):
    image_list = os.listdir(img_directory)
    random.shuffle(image_list)
    print('Number of available images:', len(image_list),'\n')
    return image_list[:num_of_test_image]

# Prepare segmentation model
sam_checkpoint =  "/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/segment-anything/checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
print('SA Used GPU:',torch.cuda.get_device_name(device=device))
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
    print('Segmentation done by SA')
    return LIPEx_segments

end_prepare = time.time()


delta_candidates =[(math.pi*7)/30,(math.pi*8)/30, (math.pi*9)/30, math.pi/3]

Jaccard_lipe_vs_lime = np.empty((0,len(delta_candidates)), int)
Jaccard_lime = np.empty((0,len(delta_candidates)), int)
Jaccard_lipe = np.empty((0,len(delta_candidates)), int)

LIME_count = np.empty((0,len(delta_candidates)), int)
LIPEx_count = np.empty((0,len(delta_candidates)), int)

TopK = 5
num_classes = 10
union_num = 3  
num_samples = 1000
batch_size=256  #256 for 3090, 1024 for A100 
num_of_test_image = 2 #?

for image_name in RandomImages(num_of_test_image):
    
    print('Image:',image_name)
    # load the image
    image_path = os.path.join(img_directory, image_name)
    image_RGB = get_image(image_path)
    pil_transf_image = pil_transf(image_RGB)
    image = np.array(pil_transf_image)
    # Test predictor for the imput image.
    pred = batch_predict([pil_transf_image])
    pred_class= pred.squeeze().argmax()
    print('Top predict class:',pred_class)

    explainer = LimeImageExplainer(random_state=42)

    sample_data, data_segments, sample_labels, sample_distances, sample_weights, LIPEx_features2use = explainer.sample_data_labels(image, 
                                                                                                                                    batch_predict,
                                                                                                                                    hide_color=0,
                                                                                                                                    num_features=union_num,
                                                                                                                                    num_samples=num_samples,
                                                                                                                                    num_exp_classes=num_classes,
                                                                                                                                    batch_size=batch_size, #256 for 3090, 1024 for A100  
                                                                                                                                    segmentation_fn=segment_image)
                                                                                                                                    

    LIPEx_exp = explainer.explain_instance_LIPEx(image,
                                                sample_data,
                                                data_segments,
                                                sample_labels,
                                                sample_distances,
                                                weights=sample_weights,
                                                used_features=LIPEx_features2use,
                                                new_top_labels=num_classes)


    # change the type of local_exp from numpy.ndarray to dict
    pred_weights = LIPEx_exp.local_exp[LIPEx_exp.top_labels[0]]
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

    LIPEx_selected_segs = [x[0] for x in LIPEx_exp.local_exp[pred_class]][:TopK]
    print('LIPEx_TopK_segments:',LIPEx_selected_segs)
    if len(LIPEx_selected_segs) < TopK:
            print ('LIPEx_selected_segs is less than',TopK)
            continue
    '''**************************End LIPEx explanation************************************'''

    LIME_exp = explainer.explain_instance_LIME(
        image,
        sample_data,
        data_segments,
        sample_labels,
        sample_distances,
        weights=sample_weights,
        true_label=[pred_class],
        num_features = len(LIPEx_features2use) 
    )

    LIME_selected_segs = [x[0] for x in LIME_exp.local_exp[pred_class]][:TopK]
    print('LIME_TopK_segments:',LIME_selected_segs)
    if len(LIME_selected_segs) < TopK:
        print ('LIME_ranked_segs is less than',TopK)
        continue

    '''**********************End LIME explanation****************************************'''

    # Compute delta-LIPEx-List-s and delta-LIME-List-s
    lipe_vs_lime =[]
    lime = []
    lipe = []
    count_documents = []
    for delta in delta_candidates:
        threshold = math.cos(delta)

        sel_indices = []
        for i in range(len(sample_data)):
            if 1 - sample_distances[i] >= threshold:
                sel_indices.append(i)
        
        # print('sel_indices size: ', len(sel_indices))
        delta_data, delta_labels, delta_distances, delta_weights = sample_data[sel_indices], sample_labels[sel_indices], sample_distances[sel_indices], sample_weights[sel_indices]
    
        print('delta_data size: {}'.format(delta_data.shape))
        
        count_documents.append(len(delta_data))

        LIPEx_exp_delta = explainer.explain_instance_LIPEx(
            image,
            delta_data,
            data_segments,
            delta_labels,
            delta_distances,
            weights=delta_weights,
            used_features=LIPEx_features2use,
            new_top_labels=num_classes,
        )
        # LIPEx
        # change the type of local_exp from numpy.ndarray to dict
        pred_weights_delta = LIPEx_exp_delta.local_exp[LIPEx_exp_delta.top_labels[0]]
        pred_sorted_indices_delta = np.argsort([np.abs(w) for w in pred_weights_delta])[::-1]
        weights_with_feature_index_delta = []
        for arr in LIPEx_exp_delta.local_exp:
            weights_with_feature_index_delta.append([(LIPEx_exp_delta.used_features[i], w) for i, w in enumerate(arr)])
        l_exp_delta = {}
        for idx, row in enumerate(weights_with_feature_index_delta):
            new_row = []
            for jdx in pred_sorted_indices_delta:
                new_row.append(row[jdx])
            l_exp_delta[idx] = new_row
        LIPEx_exp_delta.local_exp = l_exp_delta

        LIPEx_selected_segs_delta = [x[0] for x in LIPEx_exp_delta.local_exp[pred_class]][:TopK]
        print('LIPEx_selected_segments (w delta {}): {}'.format(delta, LIPEx_selected_segs_delta))
        if len(LIPEx_selected_segs_delta) < TopK:
            print ('LIPEx_selected_segments_delta is less than',TopK)
            continue

        LIME_exp_delta = explainer.explain_instance_LIME(
            image,
            delta_data,
            data_segments,
            delta_labels,
            delta_distances,
            weights=delta_weights,
            true_label=[pred_class],
            num_features=len(LIPEx_features2use),
        )

        LIME_selected_segs_delta = [x[0] for x in LIME_exp_delta.local_exp[pred_class]][:TopK]
        print('LIME_selected_segments (w delta {}): {}'.format(delta, LIME_selected_segs_delta))
        if len(LIME_selected_segs_delta) < TopK:
            print ('LIME_selected_segments_delta is less than',TopK)
            continue

        '''**********************End LIME explanation****************************************'''

        lipe_vs_lime.append(len(set(LIPEx_selected_segs_delta).intersection(set(LIME_selected_segs)))/len(set(LIPEx_selected_segs_delta).union(set(LIME_selected_segs))))
        lime.append(len(set(LIME_selected_segs_delta).intersection(set(LIME_selected_segs)))/len(set(LIME_selected_segs_delta).union(set(LIME_selected_segs))))
        lipe.append(len(set(LIPEx_selected_segs_delta).intersection(set(LIPEx_selected_segs)))/len(set(LIPEx_selected_segs_delta).union(set(LIPEx_selected_segs))))
    
    count = np.append(LIME_count, np.array([count_documents]), axis=0)
    
    Jaccard_lipe_vs_lime = np.append(Jaccard_lipe_vs_lime, np.array([lipe_vs_lime]), axis=0)
    Jaccard_lime = np.append(Jaccard_lime, np.array([lime]), axis=0)
    Jaccard_lipe = np.append(Jaccard_lipe, np.array([lipe]), axis=0)
    if len(Jaccard_lipe_vs_lime) == num_of_test_image:
        break

count = np.average(count, axis=0)


Jaccard_LIME_average = np.average(Jaccard_lime, axis=0)
Jaccard_LIPEx_average = np.average(Jaccard_lipe, axis=0)
Jaccard_LIPEx_vs_LIME_average = np.average(Jaccard_lipe_vs_lime, axis=0)


Jaccard_LIME_std = np.std(Jaccard_lime, axis=0)
Jaccard_LIPEx_std = np.std(Jaccard_lipe, axis=0)
Jaccard_LIPEx_vs_LIME_std = np.std(Jaccard_lipe_vs_lime, axis=0)

print('count sample_data:', count)


print('Jaccard_LIME_average =',Jaccard_LIME_average.tolist())
print('Jaccard_LIPEx_average =',Jaccard_LIPEx_average.tolist())
print('Jaccard_LIPEx_vs_LIME_average =',Jaccard_LIPEx_vs_LIME_average.tolist())

print('Jaccard_LIME_std =',Jaccard_LIME_std.tolist())
print('Jaccard_LIPEx_std =',Jaccard_LIPEx_std.tolist())
print('Jaccard_LIPEx_vs_LIME_std =',Jaccard_LIPEx_vs_LIME_std.tolist())


end = time.time()
print('Prepare time (s):',end_prepare - start)
print('Running time (s):',end - end_prepare)


# draw the plot
# import matplotlib.pyplot as plt
# delta_candidates_radians = [str(round(delta, ndigits=3)) for delta in delta_candidates]
# plt.figure(figsize=(20,20))
# fig, ax = plt.subplots()

# ax.errorbar(delta_candidates_radians, Jaccard_LIME_average, yerr=Jaccard_LIME_std, label=r'$J_{s,\delta,{\rm LIME}}$',capsize=4,capthick=2)
# ax.errorbar(delta_candidates_radians, Jaccard_LIPEx_average, yerr=Jaccard_LIPEx_std, label=r'$J_{s,\delta,{\rm LIPEx}}$',capsize=4,capthick=2)
# ax.errorbar(delta_candidates_radians, Jaccard_LIPEx_vs_LIME_average, yerr=Jaccard_LIPEx_vs_LIME_std, label=r'$J_{s,\delta{-}{\rm LIPEx{-}vs{-}LIME}}$',capsize=4,capthick=2)

# plt.xlabel('delta (radians)')
# plt.ylabel('Jaccard Index')
# plt.legend()
# PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/plot/jaccard/vgg16.png'
# plt.savefig(PATH)
# plt.close()



 