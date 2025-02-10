#!/usr/bin/env python
from PIL import Image
import numpy as np
import os

import torch
from torchvision import models, transforms
import torch.nn.functional as F

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_image import LimeImageExplainer

import random
random.seed(42)


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            print(img.mode)
            return img.convert('RGB')
 
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
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
model = models.vgg16()
model.classifier[6] = torch.nn.Linear(4096, 10)
model.aux_logits = False
model.load_state_dict(torch.load('/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Img/vgg16/best_ckpt/model-2.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print('GPU:',torch.cuda.get_device_name(device=device))

pil_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
    batch = batch.to(device)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

img_directory ='/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/Dataset/ImageNet/imagenette-320px/test' 

def RandomImages():
    image_list = os.listdir(img_directory)
    random.shuffle(image_list)
    print('Number of available images:', len(image_list),'\n') # 3791
    return image_list

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

LIME_count, LIPEx_count, XRAI_count, GuidedIG_count, VanillaGradient_count, SmoothGrad_count, IntegratedGradient_count = [],[],[],[],[],[],[]


TopK = 5
number_of_segments_to_remove = [1,2,3,4,5]
num_classes = 10
num_samples = 1000 # pertubation samples number

union_num = 3 
batch_size = 128 #128 for 3090, 1024 for A100
num_of_test_imgs = 3

def rank_segments(segs, scores):
    unqID,idx,IDsums = np.unique(segs,return_counts=True,return_inverse=True)
    value_sums = np.bincount(idx,scores.ravel())
    segs_socres_sum = {i:value_sums[itr] for itr,i in enumerate(unqID)}
    ranked_segs = [k for k, v in sorted(segs_socres_sum.items(), key=lambda item: item[1], reverse=True)]
    return ranked_segs # return the ranked segments sequence based on the summed scores


for image_name in RandomImages():
    print('Image:',image_name)
    # load the image
    image_path = os.path.join(img_directory, image_name)
    print('image_path:',image_path)
    im = Image.open(image_path)
    if np.asarray(im).ndim == 2:
        print('Gray image, skip')
        continue
    image_RGB = get_image(image_path)
    pil_transf_image = pil_transf(image_RGB)
    image = np.array(pil_transf_image)

    # Test predictor for the sample image.
    pred = batch_predict([pil_transf_image])
    pred_class = pred.squeeze().argmax()
    print('Top predict class:',pred_class)

    '''********************* Start LIPEx explanation *****************************************'''

    LIPEx_explainer = LimeImageExplainer(random_state=42)

    LIPEx_exp = LIPEx_explainer.explain_instance_new(image, 
                                                    batch_predict,
                                                    hide_color=0,
                                                    num_features=union_num,
                                                    num_samples=num_samples,
                                                    top_labels = num_classes,
                                                    batch_size=batch_size, 
                                                    segmentation_fn=segment_image
                                                    )
    after_union = len(LIPEx_exp.used_features)
    print('LIPEx features after union:', LIPEx_exp.used_features)

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
    LIPEx_count.append(LIPEx_count_list)

    '''*************************End LIPEx *************************************'''

    '''**************************** Start LIME explanation*******************************'''
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
    LIME_count.append(LIME_count_list)

    '''*************************End LIME *************************************'''  

    '''****************** Start Saliency ****************'''  
    image_saliency = baseline_load_image(image_path)

    for saliency in ['XRAI', 'Guided_IG', 'Vanilla_Gradient','SmoothGrad', 'Integrated_Gradient']:
        if saliency == 'XRAI':
            attributions = XRAI(image_saliency,pred_class)
        elif saliency == 'Guided_IG':
            attributions = Guided_IG(image_saliency,pred_class)
        elif saliency == 'Vanilla_Gradient':
            attributions = Vanilla_Gradient(image_saliency,pred_class)
        elif saliency == 'SmoothGrad':
            attributions = SmoothGrad(image_saliency,pred_class)
        elif saliency == 'Integrated_Gradient':
            attributions = Integrated_Gradient(image_saliency,pred_class)
        
        saliency_ranked_segs = rank_segments(segments, attributions)
        print(saliency,'TopK_segments:',saliency_ranked_segs[:TopK])

        # Saliency Re-Prediction
        saliency_count_list = rePredict(image,saliency_ranked_segs,segments,number_of_segments_to_remove)
        print(saliency,'count_list:',saliency_count_list)

        if saliency == 'XRAI':
            XRAI_count.append(saliency_count_list)
        elif saliency == 'Guided_IG':
            GuidedIG_count.append(saliency_count_list)
        elif saliency == 'Vanilla_Gradient':
            VanillaGradient_count.append(saliency_count_list)
        elif saliency == 'SmoothGrad':
            SmoothGrad_count.append(saliency_count_list)
        elif saliency == 'Integrated_Gradient':
            IntegratedGradient_count.append(saliency_count_list)
    '''******************End Saliency ****************'''

    assert len(LIPEx_count) == len(LIME_count) == len(XRAI_count)  == len(GuidedIG_count) == len(VanillaGradient_count) == len(SmoothGrad_count) == len(IntegratedGradient_count)
    if len(LIPEx_count) == num_of_test_imgs:
        break
    '''****************************End Re-Prediction**********************************'''  

LIME_array = np.array(LIME_count)
LIPEx_array = np.array(LIPEx_count)
XRAI_array = np.array(XRAI_count)
GuidedIG_array = np.array(GuidedIG_count)
Vanilla_Gradient_array = np.array(VanillaGradient_count)
SmoothGrad_array = np.array(SmoothGrad_count)
Integrated_Gradient_array = np.array(IntegratedGradient_count)


average_LIME = (LIME_array.sum(axis=0) / num_of_test_imgs).tolist()
average_LIPEx = (LIPEx_array.sum(axis=0) / num_of_test_imgs).tolist()
average_XRAI = (XRAI_array.sum(axis=0) / num_of_test_imgs).tolist()
average_GuidedIG = (GuidedIG_array.sum(axis=0) / num_of_test_imgs).tolist()
average_Vanilla_Gradient = (Vanilla_Gradient_array.sum(axis=0) / num_of_test_imgs).tolist()
average_SmoothGrad = (SmoothGrad_array.sum(axis=0) / num_of_test_imgs).tolist()
average_Integrated_Gradient = (Integrated_Gradient_array.sum(axis=0) / num_of_test_imgs).tolist()

print('LIME=',average_LIME)
print('LIPEx=',average_LIPEx)
print('XRAI=',average_XRAI)
print('GuidedIG=',average_GuidedIG)
print('Vanilla_Gradient=',average_Vanilla_Gradient)
print('SmoothGrad=',average_SmoothGrad)
print('Integrated_Gradient=',average_Integrated_Gradient)

