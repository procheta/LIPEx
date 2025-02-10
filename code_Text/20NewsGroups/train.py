import warnings
warnings.filterwarnings("ignore") # this will disable the display of all warning messages.

import torch, os, shutil
import numpy as np
import math

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
#from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt

newsgroups = load_dataset('SetFit/20_newsgroups')

num_of_train = newsgroups["train"].num_rows
slice = round(num_of_train*0.8)

newsgroups.set_format(type="pandas")
train_df = newsgroups['train'][:slice]
valid_df = newsgroups['train'][slice:]
test_df = newsgroups['test'][:]


PRETRAINED_LM = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM, do_lower_case=True)


def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=512, padding=True, return_attention_mask=True, truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

train_input_ids, train_att_masks = encode(train_df['text'].values.tolist())
valid_input_ids, valid_att_masks = encode(valid_df['text'].values.tolist())
test_input_ids, test_att_masks = encode(test_df['text'].values.tolist())


train_y = torch.LongTensor(train_df['label'].values.tolist())
valid_y = torch.LongTensor(valid_df['label'].values.tolist())
test_y = torch.LongTensor(test_df['label'].values.tolist())


BATCH_SIZE = 16 # 32 for 3090, 128 for A100

train_dataset = TensorDataset(train_input_ids, train_att_masks, train_y)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_dataset = TensorDataset(valid_input_ids, valid_att_masks, valid_y)
valid_sampler = SequentialSampler(valid_dataset)
valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input_ids, test_att_masks, test_y)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

N_labels = len(train_df.label.unique())

model = BertForSequenceClassification.from_pretrained(PRETRAINED_LM,
                                                      num_labels=N_labels,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device name:', torch.cuda.get_device_name(device=device))
model = model.to(device)



EPOCHS = 50
LEARNING_RATE = 5e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*EPOCHS )

train_loss_per_epoch = []
val_loss_per_epoch = []

PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Text/20NewsGroups'
# PATH = '/mnt/iusers01/fatpou01/compsci01/d04065hz/scratch/Hongbo/LIPEx/code_Text/20NewsGroups/'
best_model_path = os.path.join(PATH, 'best_ckpt')
_acc = float('-inf')
patience = 0

for epoch_num in range(EPOCHS):
    print('Epoch: ', epoch_num + 1)
    '''
    Training
    '''
    model.train()
    train_loss = 0
    for step_num, batch_data in enumerate(tqdm(train_dataloader,desc='Training')):
        input_ids, att_mask, labels = [data.to(device) for data in batch_data]
        output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)
        
        loss = output.loss
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
    valid_acc = []
 
    with torch.no_grad():
        for step_num_e, batch_data in enumerate(tqdm(valid_dataloader,desc='Validation')):
            input_ids, att_mask, labels = [data.to(device) for data in batch_data]
            output = model(input_ids = input_ids, attention_mask=att_mask, labels= labels)

            loss = output.loss
            valid_loss += loss.item()

            step_output = output.logits
            step_output = torch.softmax(step_output, dim=1).detach().cpu().numpy()
            step_output = np.argmax(step_output, axis=1).flatten()
            step_label_ids = labels.cpu().numpy().flatten()
            if step_num_e == 0:
                y_pred = step_output
                y_true = step_label_ids
            else:
                y_pred = np.concatenate((y_pred, step_output))
                y_true = np.concatenate((y_true, step_label_ids))
        
        accuracy = (y_true == y_pred).sum() / len(y_true)
        valid_acc.append(accuracy)
        print("Validation Accuracy: {}".format(accuracy))
        if accuracy > _acc:
            patience = 0
            _acc = accuracy
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
                os.makedirs(best_model_path)
            else:
                os.makedirs(best_model_path)
            torch.save(model.state_dict(), os.path.join(best_model_path, 'model-'+str(epoch_num+1)+'.pt'))
        else:
            patience += 1
            if patience > 4:
                break

    valid_loss = (valid_loss / (step_num_e + 1))
    val_loss_per_epoch.append(valid_loss)
        
    '''
    Loss message
    '''
    print("{0}/{1} train loss: {2} ".format(step_num+1, math.ceil(len(train_df) / BATCH_SIZE), train_loss / (step_num + 1)))
    print("{0}/{1} val loss: {2} ".format(step_num_e+1, math.ceil(len(valid_df) / BATCH_SIZE), valid_loss / (step_num_e + 1)))


# draw the loss curve
epochs = range(1, epoch_num +1 )
fig, ax = plt.subplots()
ax.plot(epochs,train_loss_per_epoch,label ='training loss')
ax.plot(epochs, val_loss_per_epoch, label = 'validation loss' )
ax.plot(epochs, valid_acc, label = 'validation accuracy' )
ax.set_title('Training and Validation loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
plt.savefig(PATH+'/loss-'+str(epoch_num+1)+'.png')
plt.close()