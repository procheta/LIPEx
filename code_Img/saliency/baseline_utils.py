'''
Script for baseline models.
Copied from https://github.com/PAIR-code/saliency/blob/master/Examples_pytorch.ipynb
'''
# Boilerplate imports.
import numpy as np
import PIL.Image
import torch
from torchvision import models, transforms

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((224, 224))
    # im = im.crop((16, 16, 240, 240))
    im = np.asarray(im)
    return im

transformer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def PreprocessImages(images):
    # assumes input is 4-D, with range [0,255]
    #
    # torchvision have color channel as first dimension
    # with normalization relative to mean/std of ImageNet:
    #    https://pytorch.org/vision/stable/models.html
    images = np.array(images)
    images = images/255
    images = np.transpose(images, (0,3,1,2))
    images = torch.tensor(images, dtype=torch.float32)
    images = transformer.forward(images)
    return images.requires_grad_(True)

# # Loading the InceptionV3 model for ImageNet
# model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
model = models.vit_b_16()
model.heads[0] = torch.nn.Linear(768, 10)
model.aux_logits = False
# model.load_state_dict(torch.load('/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Img/vit/model-3.pt'))
eval_mode = model.eval()

# Register hooks for Grad-CAM, which uses the last convolution layer
# conv_layer = model.Mixed_7c
# conv_layer_outputs = {}

# def conv_layer_forward(m, i, o):
#     # move the RGB dimension to the last dimension
#     conv_layer_outputs[saliency.base.CONVOLUTION_LAYER_VALUES] = torch.movedim(o, 1, 3).detach().numpy()
# def conv_layer_backward(m, i, o):
#     # move the RGB dimension to the last dimension
#     conv_layer_outputs[saliency.base.CONVOLUTION_OUTPUT_GRADIENTS] = torch.movedim(o[0], 1, 3).detach().numpy()

# conv_layer.register_forward_hook(conv_layer_forward)
# conv_layer.register_full_backward_hook(conv_layer_backward)


class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    images = PreprocessImages(images)
    target_class_idx =  call_model_args[class_idx_str]
    output = model(images)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = output[:,target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.detach().numpy()
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

def baseline_load_image(image_path):
    # Load the image
    im_orig = LoadImage(image_path)
    im = im_orig.astype(np.float32)
    return im

def Vanilla_Gradient(im, prediction_class):
    call_model_args = {class_idx_str: prediction_class}
    # Construct the saliency object. This alone doesn't do anthing.
    gradient_saliency = saliency.GradientSaliency()

    # Compute the vanilla gradient. This is a fast approximation method.
    vanilla_mask_3d = gradient_saliency.GetMask(im, call_model_function, call_model_args)
    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    return vanilla_mask_grayscale

def SmoothGrad(im, prediction_class):
    call_model_args = {class_idx_str: prediction_class}
    # Construct the saliency object. This alone doesn't do anthing.
    gradient_saliency = saliency.GradientSaliency()

    # Compute the vanilla gradient. This is a fast approximation method.
    smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, call_model_function, call_model_args)
    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    return smoothgrad_mask_grayscale

def XRAI(im, prediction_class):
    call_model_args = {class_idx_str: prediction_class}
    # Construct the saliency object. This alone doesn't do anthing.
    xrai_object = saliency.XRAI()

    # Compute XRAI attributions with default parameters
    xrai_attributions = xrai_object.GetMask(im, call_model_function, call_model_args, batch_size=20)
    return xrai_attributions

def Guided_IG(im,prediction_class):
    call_model_args = {class_idx_str: prediction_class}
    # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
    guided_ig = saliency.GuidedIG()

    # Baseline is a black image for vanilla integrated gradients.
    baseline = np.zeros(im.shape)

    # Compute the Guided IG mask.
    guided_ig_mask_3d = guided_ig.GetMask(
        im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    guided_ig_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)

    return guided_ig_mask_grayscale

def Integrated_Gradient(im, prediction_class):
    call_model_args = {class_idx_str: prediction_class}
    # Construct the saliency object. This alone doesn't do anthing.
    integrated_gradient = saliency.IntegratedGradients()

    # Baseline is a black image for vanilla integrated gradients.
    baseline = np.zeros(im.shape)

    # Compute the vanilla gradient. This is a fast approximation method.
    integrated_gradients_mask_3d = integrated_gradient.GetMask(
        im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    integrated_gradients_mask_grayscale = saliency.VisualizeImageGrayscale(integrated_gradients_mask_3d)
    return integrated_gradients_mask_grayscale

def Blur_IG(im,prediction_class):
    call_model_args = {class_idx_str: prediction_class}
    # Construct the saliency object. This alone doesn't do anthing.
    blur_ig = saliency.BlurIG()

    # Compute the Blur IG mask.
    blur_ig_mask_3d = blur_ig.GetMask(im, call_model_function, call_model_args, batch_size=20)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)
    return blur_ig_mask_grayscale


