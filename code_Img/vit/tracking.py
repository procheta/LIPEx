#!/usr/bin/env python
from PIL import Image
import numpy as np
import os

import torch
from torchvision import models, transforms
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import sys
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
    with torch.inference_mode():
        logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

img_directory ='/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/dataset/ImageNet/imagenette-320px/test'

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
    print('Segmentation done by SA')
    return LIPEx_segments

def f_rePredict(image,TopK_segments,segments):
    copy_image = np.copy(image)
    for i in range(len(TopK_segments)):
        removed_segment = TopK_segments[i]
        copy_image[segments == removed_segment] = 0
    f_re_pred = batch_predict([copy_image])
    f_repred_class = f_re_pred.squeeze().argmax()
    return f_repred_class

def exp_rePredict(sample_data,explainer_TopK_segments,explainer_exp,explainer_model):
    # Re-Prediction of explainer
    new_sample_data = sample_data[0].copy()
    new_sample_data[explainer_TopK_segments] = 0
    new_sample_data = torch.from_numpy(new_sample_data[explainer_exp.used_features]).float()
    explainer_predict = explainer_model.predict(new_sample_data.reshape(1,-1))
    _, explainer_predicted = torch.max(explainer_predict.data, 1)
    explainer_repred_label= explainer_predicted.detach().numpy()[0] #  Predicted top label
    return explainer_repred_label


TopK = 5
number_of_segments_to_remove = [1,2,3,4,5]
num_classes = 10
union_num = 3
num_samples = 1000

batch_size = 256
num_of_test_image = 100

LIPEx_list = []
for i in range(3):
    LIPEx_count = []
    for image_name in RandomImages():
        LIPEx_count_list = []
        print('Image:',image_name)
        # load the image
        image_path = os.path.join(img_directory, image_name)
        image_RGB = get_image(image_path)
        pil_transf_image = pil_transf(image_RGB)
        image = np.array(pil_transf_image)
        # Test predictor for the sample image.
        pred = batch_predict([pil_transf_image])
        pred_class = pred.squeeze().argmax()
        print('Top predict class:',pred_class)
        
        # LIPEx
        LIPEx_explainer = LimeImageExplainer(random_state=42)

        sample_data, data_segments, sample_labels, sample_distances, sample_weights, LIPEx_features2use = LIPEx_explainer.sample_data_labels(image, 
                                                                                                                                            batch_predict,
                                                                                                                                            hide_color=0,
                                                                                                                                            num_features=union_num,
                                                                                                                                            num_samples=num_samples,
                                                                                                                                            num_exp_classes=num_classes,
                                                                                                                                            batch_size=batch_size, 
                                                                                                                                            segmentation_fn=segment_image,
                                                                                                                                            )

        LIPEx_exp = LIPEx_explainer.explain_instance_LIPEx(image,
                                                            sample_data,
                                                            data_segments,
                                                            sample_labels,
                                                            sample_distances,
                                                            weights=sample_weights,
                                                            used_features=LIPEx_features2use,
                                                            new_top_labels=num_classes)

        print('LIPEx_exp.used_features:', LIPEx_exp.used_features)

        # reranking the LIPEx_exp.local_exp according to the top prediect class
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

        # check if the number of segments is no less than TopK
        LIPEx_ranked_segs = [x[0] for x in LIPEx_exp.local_exp[pred_class]][:TopK]
        print('LIPEx_TopK_segments:',LIPEx_ranked_segs)
        if len(LIPEx_ranked_segs) < TopK:
            print ('LIPEx_selected_segs is less than',TopK)
            continue

        segments = LIPEx_exp.segments

        # Re-Prediction
        for num in number_of_segments_to_remove:
            f_repred_label = f_rePredict(image,LIPEx_ranked_segs[:num],segments)
            LIPEx_repred_label = exp_rePredict(sample_data,LIPEx_ranked_segs[:num],LIPEx_exp,LIPEx_explainer.base.LIPEx_model)
            print('f_repred_label:',f_repred_label,'LIPEx_repred_label:',LIPEx_repred_label)
            if f_repred_label == LIPEx_repred_label:
                LIPEx_count_list.append(1)
            else:
                LIPEx_count_list.append(0)
        LIPEx_count.append(LIPEx_count_list)
        if len(LIPEx_count) == num_of_test_image:
            break

    LIPEx_list.append(np.average(np.array(LIPEx_count),axis=0).tolist())

LIPEx_array = np.array(LIPEx_list)

average_LIPEx = np.average(LIPEx_array,axis=0)
std_LIPEx = np.std(LIPEx_array,axis=0)

print('average_LIPEx =',average_LIPEx.tolist())
print('std_LIPEx =',std_LIPEx.tolist())



# LIPEx_count = []
# for image_name in RandomImages():
#     LIPEx_count_list = []
#     print('Image:',image_name)
#     # load the image
#     image_path = os.path.join(img_directory, image_name)
#     image_RGB = get_image(image_path)
#     pil_transf_image = pil_transf(image_RGB)
#     image = np.array(pil_transf_image)
#     # Test predictor for the sample image.
#     pred = batch_predict([pil_transf_image])
#     pred_class = pred.squeeze().argmax()
#     print('Top predict class:',pred_class)
    
#     # LIPEx
#     LIPEx_explainer = LimeImageExplainer(random_state=42)

#     sample_data, data_segments, sample_labels, sample_distances, sample_weights, LIPEx_features2use = LIPEx_explainer.sample_data_labels(image, 
#                                                                                                                                         batch_predict,
#                                                                                                                                         hide_color=0,
#                                                                                                                                         num_features=union_num,
#                                                                                                                                         num_samples=num_samples,
#                                                                                                                                         num_exp_classes=num_classes,
#                                                                                                                                         batch_size=batch_size, 
#                                                                                                                                         segmentation_fn=segment_image,
#                                                                                                                                         )

#     LIPEx_exp = LIPEx_explainer.explain_instance_LIPEx(image,
#                                                         sample_data,
#                                                         data_segments,
#                                                         sample_labels,
#                                                         sample_distances,
#                                                         weights=sample_weights,
#                                                         used_features=LIPEx_features2use,
#                                                         new_top_labels=num_classes)

#     print('LIPEx_exp.used_features:', LIPEx_exp.used_features)

#     # reranking the LIPEx_exp.local_exp according to the top prediect class
#     pred_weights = LIPEx_exp.local_exp[pred_class]
#     pred_sorted_indices = np.argsort([np.abs(w) for w in pred_weights])[::-1]
#     weights_with_feature_index = []
#     for arr in LIPEx_exp.local_exp:
#         weights_with_feature_index.append([(LIPEx_exp.used_features[i], w) for i, w in enumerate(arr)])
#     l_exp = {}
#     for idx, row in enumerate(weights_with_feature_index):
#         new_row = []
#         for jdx in pred_sorted_indices:
#             new_row.append(row[jdx])
#         l_exp[idx] = new_row
#     LIPEx_exp.local_exp = l_exp

#     # check if the number of segments is no less than TopK
#     LIPEx_ranked_segs = [x[0] for x in LIPEx_exp.local_exp[pred_class]][:TopK]
#     print('LIPEx_TopK_segments:',LIPEx_ranked_segs)
#     if len(LIPEx_ranked_segs) < TopK:
#         print ('LIPEx_selected_segs is less than',TopK)
#         continue

#     segments = LIPEx_exp.segments

#     # Re-Prediction
#     for num in number_of_segments_to_remove:
#         f_repred_label = f_rePredict(image,LIPEx_ranked_segs[:num],segments)
#         LIPEx_repred_label = exp_rePredict(sample_data,LIPEx_ranked_segs[:num],LIPEx_exp,LIPEx_explainer.base.LIPEx_model)
#         print('f_repred_label:',f_repred_label,'LIPEx_repred_label:',LIPEx_repred_label)
#         if f_repred_label == LIPEx_repred_label:
#             LIPEx_count_list.append(1)
#         else:
#             LIPEx_count_list.append(0)
#     LIPEx_count.append(LIPEx_count_list)
#     if len(LIPEx_count) == num_of_test_image:
#         break

# average_LIPEx = np.average(np.array(LIPEx_count),axis=0)
# print('average_LIPEx =',average_LIPEx.tolist())
