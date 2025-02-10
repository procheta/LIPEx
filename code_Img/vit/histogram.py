#!/usr/bin/env python
from PIL import Image
import numpy as np
import os

import torch
from torchvision import models, transforms
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_image import LimeImageExplainer

import random
# random.seed(42)
# import time
# strat_time = time.time()


#  load fine-tuned model
model = models.vit_b_16(weights='DEFAULT')
model.heads[0] = torch.nn.Linear(768, 10)
model.aux_logits = False
model.load_state_dict(torch.load('/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Img/vit/model-3.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))

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

pil_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

#Now we are ready to define classification function.
#The input to this function is numpy array of images where each image is ndarray of shape **(channel, height, width)**. 
#The output is numpy array of shape (image index, classes) where each value in array should be probability for that image, class combination.

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    model.to(device)
    batch = batch.to(device)
    with torch.inference_mode():
        logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

img_directory ='/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/dataset/ImageNet/imagenette-320px/test' # 3925 

def RandomImages(num_of_test_image):
    image_list = os.listdir(img_directory)
    random.shuffle(image_list)
    print('Number of available images:', len(image_list),'\n')
    return image_list[:num_of_test_image]

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
    print('Segmentation done by SA')
    return LIPEx_segments

# distance metrics of two distributions
def TV(p, q):
        TotalVar_D = torch.mean(0.5 * (torch.sum(torch.abs(p-q), dim=-1))) 
        return TotalVar_D.item()


num_classes = 10
batch_size = 64 
union_num = 3  
num_samples = 1000 

num_of_test_image = 3 #?

TV_s = [] 
for image_name in RandomImages(num_of_test_image):
    
    print('Image:',image_name)
    # load the image
    image_path = os.path.join(img_directory, image_name)
    image_RGB = get_image(image_path)
    pil_transf_image = pil_transf(image_RGB)
    image = np.array(pil_transf_image)
    # Test predictor for the imput image.
    pred = batch_predict([pil_transf_image])
    
    # LIPEx
    LIPEx_explainer = LimeImageExplainer(random_state=42)

    sample_data, data_segments, sample_labels, sample_distances, sample_weights, LIPEx_features2use = LIPEx_explainer.sample_data_labels(image, 
                                                                                                                                        batch_predict,
                                                                                                                                        hide_color=0,
                                                                                                                                        num_features=union_num,
                                                                                                                                        num_samples=num_samples,
                                                                                                                                        num_exp_classes=num_classes,
                                                                                                                                        batch_size=batch_size, 
                                                                                                                                        segmentation_fn=segment_image
                                                                                                                                        )

    LIPEx_exp = LIPEx_explainer.explain_instance_LIPEx(image,
                                                        sample_data,
                                                        data_segments,
                                                        sample_labels,
                                                        sample_distances,
                                                        weights=sample_weights,
                                                        used_features=LIPEx_features2use,
                                                        new_top_labels=num_classes)

    print('TV_distance',TV(torch.from_numpy(pred),LIPEx_exp.local_pred))
    TV_s.append(TV(torch.from_numpy(pred),LIPEx_exp.local_pred))

print('\n')
print('TV(f(s), LIPEx(s)) =',TV_s)
print('len(TV_s)) =',len(TV_s))

print('Avg(TV(f(s), LIPEx(s))) =',np.mean(TV_s))
print('Std(TV(f(s), LIPEx(s))) =',np.std(TV_s))




