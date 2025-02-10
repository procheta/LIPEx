# prepare images
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import random
random.seed(42)
import os, json


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
    # print('Number of available images:', len(chosenImage))
    # chosenImage = allImages[choice] #Do Whatever you want with the image file
    return [os.path.join(directory, img) for img in chosenImage]

randomImage = chooseRandomImage()
print('Number of available images:', len(randomImage))

# prepare SegmentAnything
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/segment-anything/checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"
print(torch.cuda.get_device_name(device=device))
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# apply SegmentAnything on images with more than 100 segments are found
from tqdm import tqdm
valid_images, images_segments_num, segmented_images = [], [], []
for image_path in tqdm(randomImage):
    image = cv2.imread(image_path)
    image_name = image_path.split('/')[-1]
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print("Image shape:", image.shape)
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=(lambda x: x['area']))

    LIPEx_segments = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for idx in range(len(masks))[::-1]:
        cur_mask = masks[idx]['segmentation']
        LIPEx_segments[cur_mask] = idx
    # np.save(os.path.join('ImageNette_Segments', image_name.replace('.JPEG', 'npy')), LIPEx_segments)
    np.save(os.path.join('ImageNette_Segments', image_name.replace('.JPEG', '')), LIPEx_segments)
    num_seg = len(np.unique(LIPEx_segments))
    segmented_images.append(image_path)
    with open('ImageNette/ImageNette_Segmented_files.txt', 'a') as sf:
         sf.write('{},{}\n'.format(image_name, num_seg))
    images_segments_num.append(num_seg)
    if num_seg > 99:
           valid_images.append(image_path)
           with open('ImageNette/valid_images.txt', 'a') as f:
               f.write(image_name+'\n')