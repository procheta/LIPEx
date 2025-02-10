#!/usr/bin/env python
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import copy

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_image import LimeImageExplainer
import random

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))

#  load fine-tuned model
model = models.vgg16()
model.classifier[6] = torch.nn.Linear(4096, 10)
model.load_state_dict(torch.load('/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Img/vgg16/best_ckpt/model-2.pt'))

cls2idx = {'n01440764': 0, 'n02102040': 1, 'n02979186': 2, 'n03000684': 3, 'n03028079': 4, 'n03394916': 5, 'n03417042': 6, 'n03425413': 7, 'n03445777': 8, 'n03888257': 9}

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
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

pil_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

class Predictor(object):
    def __init__(self, model):
        self.model = model.to(device)
        self.model.eval()

    def predict(self, images):
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        batch = batch.to(device)
        with torch.inference_mode():
            logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
    
p = Predictor(model)

img_directory ='/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/dataset/ImageNet/imagenette-320px/test' 

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

def accuracy(Predictor, test_data_set, true_idx_list):
    pred = Predictor.predict(test_data_set)
    pred = np.argmax(pred, axis=1)
    pred_idx_list = pred.tolist()
    acc = sum([1 if pred_idx_list[i]==true_idx_list[i] else 0 for i in range(len(pred))])/len(pred)
    return acc

def gaussian_noise(x, var):
  return torch.normal(0, var, size=x.shape).to(x.device)

def add_noise(layer, var):
    with torch.no_grad():
        for _, param in layer.named_parameters():
                param.add_(gaussian_noise(param, var))
        return
    
# distance metrics of two distributions
def TV(p, q):
        TotalVar_D = torch.mean(0.5 * (torch.sum(torch.abs(p-q), dim=-1))) 
        return TotalVar_D.item()

num_classes = 10
union_num = 3 
num_samples = 1000
batch_size = 128
test_imgs_num = 10 # need to be defined

image_list = RandomImages(test_imgs_num)

test_data_set = [pil_transf(get_image(os.path.join(img_directory, image_name))) for image_name in image_list]
true_idx_list = [cls2idx[image_name.split('_')[0]] for image_name in image_list]

model_TV_average_list, LIPEx_TV_average_list = [], []
var_list = [round(var,2) for var in np.arange(0, 0.3, 0.03).tolist()] # to be defined
accuracy_list = []


for var in var_list:
    copy_model = copy.deepcopy(model)
    add_noise(copy_model.classifier[6], var)
    distort_p = Predictor(copy_model)
    accuracy_list.append(accuracy(distort_p, test_data_set, true_idx_list))

    model_TV_distance = []
    LIPEx_TV_distance = []
    
    for image_name in image_list:
        
        # print('image_name:',image_name)

        image_path = os.path.join(img_directory, image_name)
        
        pil_transf_image = pil_transf(get_image(image_path))

        image = np.array(pil_transf_image)
        
        pred = p.predict([pil_transf_image])

        distort_pred = distort_p.predict([pil_transf_image])

        # compute TV distance of two model output distributions

        model_TV_distance.append(TV(torch.from_numpy(pred), torch.from_numpy(distort_pred)))

        # LIPEx_explainer = LimeImageExplainer(random_state=42)

        # #------------------------Below for original model------------------------#
        # LIPEx_exp = LIPEx_explainer.explain_instance_new(image, 
        #                                                 p.predict, # multiclass classification function
        #                                                 hide_color = 0,
        #                                                 top_labels = num_classes,
        #                                                 num_features = union_num,
        #                                                 num_samples = num_samples,
        #                                                 batch_size=batch_size, 
        #                                                 segmentation_fn = segment_image) 
        # local_pred = LIPEx_exp.local_pred

        # #------------------------Below for distorted model------------------------#
        

        # distort_LIPEx_exp = LIPEx_explainer.explain_instance_new(image, 
        #                                                         distort_p.predict, # multiclass classification function
        #                                                         hide_color = 0,
        #                                                         top_labels = num_classes,
        #                                                         num_features = union_num,
        #                                                         num_samples = num_samples,
        #                                                         batch_size=batch_size, 
        #                                                         segmentation_fn = segment_image) 
        # distort_local_pred = distort_LIPEx_exp.local_pred
        #  # compute TV distance of two LIPEx output distributions
        # LIPEx_TV_distance.append(TV(local_pred, distort_local_pred))
        
    model_TV_distance = np.array(model_TV_distance)
    model_TV_average = np.mean(model_TV_distance, axis=0)
    model_TV_average_list.append(model_TV_average)

    # LIPEx_TV_distance = np.array(LIPEx_TV_distance)
    # LIPEx_TV_average = np.mean(LIPEx_TV_distance, axis=0)
    # LIPEx_TV_average_list.append(LIPEx_TV_average)


print('var_list =',var_list)
print('accuracy_list =',accuracy_list)
print('model_TV_average_list =',model_TV_average_list)
# print('LIPEx_TV_average_list =',LIPEx_TV_average_list)
