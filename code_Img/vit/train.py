#!/usr/bin/env python
from PIL import Image
import numpy as np
import os, shutil, math

import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt

#  configuration
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 5e-5
PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Img/vit' 
# PATIENCE = 2


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
}
# load images
img_directory="/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/Dataset/ImageNet/imagenette-320px"
dsets = {
    x: datasets.ImageFolder(os.path.join(img_directory, x), data_transforms[x]) for x in ['train', 'val']
}

# split the val set from train set and take the original val set as test set
# train_idx, val_idx = train_test_split(list(range(len(dsets['train']))), test_size=0.2, random_state=42)
# new_dsets = {
#     'train': Subset(dsets['train'], train_idx),
#     'val': Subset(dsets['train'], val_idx),
#     'test': dsets['val']
# }
# print(f"Number of training/validation/test examples: {len(new_dsets['train'])}\t{len(new_dsets['val'])}\t{len(new_dsets['test'])}")

print(f"Number of training/validation examples: {len(dsets['train'])}\t{len(dsets['val'])}")

train_dataloader = torch.utils.data.DataLoader(dsets['train'], batch_size = BATCH_SIZE, shuffle=True, num_workers=4)
valid_dataloader = torch.utils.data.DataLoader(dsets['val'], batch_size = BATCH_SIZE, num_workers=4)
# test_dataloader = torch.utils.data.DataLoader(new_dsets['test'], batch_size = BATCH_SIZE, num_workers=4)

num_train_batches = len(train_dataloader)
num_valid_batches = len(valid_dataloader)

# print(f"Number of training/validation/test batches: {len(train_dataloader)}\t{len(valid_dataloader)}\t{len(test_dataloader)}")
print(f"Number of training/validation batches: {len(train_dataloader)}\t{len(valid_dataloader)}")
dset_classes = dsets['train'].classes

# load the pre-trained model
model = models.vit_b_16(weights='DEFAULT')
model.heads[0] = torch.nn.Linear(768, 10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))
model = model.to(device)

# training loop
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*EPOCHS )
criterion = torch.nn.CrossEntropyLoss()

train_loss_per_epoch = []
# val_loss_per_epoch = []
# _acc = float('-inf')
# patience_cnt = 0
# best_model_path = os.path.join(PATH, 'best_ckpt')

for epoch_num in range(EPOCHS):
    print('Epoch: ', epoch_num + 1)
    '''
    Training
    '''
    model.train()
    train_loss = 0
    for step_num, batch_data in enumerate(tqdm(train_dataloader,desc='Training')):
        inputs, labels = [data.to(device) for data in batch_data]
        outputs = model(inputs)
        
        if isinstance(outputs, tuple):
            loss = sum((criterion(o, labels) for o in outputs))
        else:
            loss = criterion(outputs, labels)
        train_loss += loss.item()

        model.zero_grad()
        loss.backward()
        del loss

        clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    train_loss_per_epoch.append(train_loss / (step_num + 1))

'''
Validation
'''
model.eval()
valid_loss = 0
# valid_acc = []

with torch.no_grad():
    for step_num_e, batch_data in enumerate(tqdm(valid_dataloader,desc='Validation')):
        inputs, labels = [data.to(device) for data in batch_data]
        outputs = model(inputs)

        if isinstance(outputs, tuple):
            loss = sum((criterion(o, labels) for o in outputs))
        else:
            loss = criterion(outputs, labels)
        valid_loss += loss.item()

        # print(f"outputs with shape {outputs.shape}: {outputs}")
        if isinstance(outputs, tuple):
            # inception v3 output will be (x, aux)
            outputs = outputs[0]
        step_output = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        step_output = np.argmax(step_output, axis=1).flatten()
        step_label_ids = labels.cpu().numpy().flatten()
        if step_num_e == 0:
            y_pred = step_output
            y_true = step_label_ids
        else:
            y_pred = np.concatenate((y_pred, step_output))
            y_true = np.concatenate((y_true, step_label_ids))
    
    accuracy = (y_true == y_pred).sum() / len(y_true)
    # valid_acc.append(accuracy)
    print("Validation Accuracy: {}".format(accuracy))
    # if accuracy > _acc:
    #     patience_cnt = 0
    #     _acc = accuracy
    #     if os.path.exists(best_model_path):
    #         shutil.rmtree(best_model_path)
    #         os.makedirs(best_model_path)
    #     else:
    #         os.makedirs(best_model_path)
    #     torch.save(model.state_dict(), os.path.join(best_model_path, 'model-'+str(epoch_num+1)+'.pt'))
    # else:
    #     patience_cnt += 1
    #     if patience_cnt > PATIENCE-1:
    #         break

valid_loss = (valid_loss / (step_num_e + 1))
# val_loss_per_epoch.append(valid_loss)
    
'''
Loss message
'''
print("{0}/{1} train loss: {2} ".format(step_num+1, num_train_batches, train_loss / (step_num + 1)))
print("{0}/{1} val loss: {2} ".format(step_num_e+1, num_valid_batches, valid_loss / (step_num_e + 1)))

torch.save(model.state_dict(), os.path.join(PATH, 'model-'+str(epoch_num+1)+'.pt'))              



# # draw the loss curve
# epochs = range(1, EPOCHS +1 )
# fig, ax = plt.subplots()
# print(f"epochs: {epochs}")
# print(f"train_loss_per_epoch: {train_loss_per_epoch}")
# ax.plot(epochs,train_loss_per_epoch,label ='training loss')
# # ax.plot(epochs, val_loss_per_epoch, label = 'validation loss' )
# # ax.plot(epochs, valid_acc, label = 'validation accuracy' )
# ax.set_title('Training and Validation loss')
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.legend()
# plt.savefig(PATH+'/loss-'+str(epoch_num+1)+'.png')
# plt.close()