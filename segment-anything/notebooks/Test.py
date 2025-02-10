import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import random
random.seed(42)
import os, json

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0 
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]]) # random color with alpha=0.35
        img[m] = color_mask
    ax.imshow(img)

# test_imgs_idx = random.sample(range(0, len(ImgeNet)), test_imgs_num)
imgExtension = ["png", "jpeg", "JPEG", "jpg"] #Image Extensions to be chosen from
allImages = list()
def chooseRandomImage(directory="/media/hongbo/Experiment/Hongbo/ImageNet/imagenette-320px/val"):
    for img in os.listdir(directory): #Lists all files
        ext = img.split(".")[-1]
        if (ext in imgExtension):
            allImages.append(img)
    # choice = random.randint(0, len(allImages) - 1)
    # chosenImage = random.choices(allImages, k = test_imgs_num)
    random.shuffle(allImages)
    chosenImage = list(set(allImages))
    print('Number of available images:', len(chosenImage))
    # chosenImage = allImages[choice] #Do Whatever you want with the image file
    return [os.path.join(directory, img) for img in chosenImage]

randomImage = chooseRandomImage()
print('Number of chosen images:', len(randomImage))
image_path = randomImage[random.randint(0, len(randomImage)-1)]
print(image_path)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image shape:", image.shape)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/segment-anything/checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
print(torch.cuda.get_device_name(device=device))
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)


LIPEx_segments = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
for idx in range(len(masks)):
    assert (LIPEx_segments).shape == (masks[idx]['segmentation'].astype(np.uint8)).shape
    LIPEx_segments+=(masks[idx]['segmentation'].astype(np.uint8))*idx
print(LIPEx_segments)
print(len(np.unique(LIPEx_segments)), np.unique(LIPEx_segments))

