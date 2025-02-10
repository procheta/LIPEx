import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from datasets import load_dataset

from sklearn.utils import Bunch
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_text import LimeTextExplainer


emotion = load_dataset('emotion')

# test_text, test_label = emotion['test']['text'], emotion['test']['label']
# new_test_text, new_text_label = [], []

# for i in range(len(test_text)):
#     if len(set(test_text[i].split(' '))) > 30:
#         new_test_text.append(test_text[i].lower())
#         new_text_label.append(test_label[i])
# assert len(new_test_text) == len(new_text_label)

# new_test = Bunch(data=new_test_text, target=np.array(new_text_label))
# print('Total texts in test dataset:',len(new_test.data)) 

train_text, train_label = emotion['train']['text'], emotion['train']['label']
new_train_text, new_train_label = [], []
for i in range(len(train_text)):
    if len(set(train_text[i].split(' '))) > 30: 
        new_train_text.append(train_text[i].lower())
        new_train_label.append(train_label[i])
assert len(new_train_text) == len(new_train_label)
new_train = Bunch(data=new_train_text, target=np.array(new_train_label))
print('Total texts in train dataset:',len(new_train.data)) #1327


label_names = emotion["train"].features['label'].names
print('dataset_label:',label_names,type(label_names))
N_labels = len(label_names)

PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Text/Emotion'
# PATH = '/mnt/iusers01/fatpou01/compsci01/d04065hz/scratch/Hongbo/LIPEx/code_Text/Emotion'
PRETRAINED_LM = "bert-base-uncased"

config = BertConfig.from_pretrained(PRETRAINED_LM)
config.num_labels = N_labels
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load(PATH+'/model-25.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)
def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=128, padding=True, return_attention_mask=True, truncation=True, return_tensors='pt') # max_length to be defined

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Wrap tokenizer and model for LIME
class pipeline(object):
    def __init__(self, model, encoder): 
        self.model = model.to(device)
        self.encoder = encoder
        self.model.eval()
    
    def predict(self, text, batch_size=128): #batch_size to be defined
        num_batches = int(len(text)/batch_size) if len(text)%batch_size == 0 else int(len(text)/batch_size)+1
        out = []
        for num in range(num_batches):
            batch_text = text[num*batch_size:(num+1)*batch_size]

            batch_input_ids,batch_attention_mask = self.encoder(batch_text)

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)

            batch_output = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits # (batch_size, num_class)
            batch_out = batch_output.softmax(dim=-1).cpu().detach().tolist() # (batch_size, num_class)
            out += batch_out
        return np.array(out)

c = pipeline(model,encoder=encode)

# distance metrics of two distributions
def TV(p, q):
        TotalVar_D = torch.mean(0.5 * (torch.sum(torch.abs(p-q), dim=-1))) 
        return TotalVar_D.item()

def f_re_pred_dis(idx,explainer_TopK_words):
   # Re-Prediction of predictor
    processed_text  = new_train.data[idx]
    for word in explainer_TopK_words:
        processed_text = processed_text.replace(word,'')
    f_output = c.predict([processed_text])
    return torch.from_numpy(f_output)

def exp_re_pred_dis(sample_data,explainer_TopK_features_idx,explainer_exp,explainer_model):
    # Re-Prediction of explainer
    new_sample_data = sample_data[0].copy()
    new_sample_data[explainer_TopK_features_idx] = 0
    new_sample_data = torch.from_numpy(new_sample_data[explainer_exp.used_features]).float()
    explainer_repred = explainer_model.predict(new_sample_data.reshape(1,-1))
    return explainer_repred

def random_chose_documents(dataset,num_of_documents):
    random_documents_idx = random.sample(range(0, len(dataset.data)), num_of_documents)
    return random_documents_idx

TopK = 5 
number_of_features_to_remove = [1,2,3,4,5]

union_num = 5
num_samples = 1000 # to be defined
num_of_test_documents = 3 # to be defined

remove_TV_distance = []
original_TV_distance = [] 

for idx in random_chose_documents(new_train,num_of_test_documents):

    f_output = c.predict([new_train.data[idx]]) # (num_text, num_class)
    _, predicted = torch.max(torch.tensor(f_output), 1)
    pred_label= predicted.detach().numpy()[0] #  get the Predicted top label index

    print('Document ID:',idx,',','f_Predicted label:',label_names[pred_label],',','True label:',label_names[new_train.target[idx]]) 
    
    #------------------------Below for  LIPEx ------------------------#
    explainer = LimeTextExplainer(class_names=label_names,random_state=42)
    # sample perturbation data, features2use: Union Set 
    sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_train.data[idx], c.predict, num_features=union_num, num_samples=num_samples)

    LIPEx_exp = explainer.explain_instance_LIPEx(
        sample_data,
        sample_labels,
        sample_distances,
        sample_weights,
        used_features=features2use,
        new_top_labels=N_labels
    )
    print('original_diatance',TV(torch.from_numpy(f_output),LIPEx_exp.local_pred))
    original_TV_distance.append(TV(torch.from_numpy(f_output),LIPEx_exp.local_pred))
   
    # LIPEx
    LIPEx_TopK_features_idx=[LIPEx_exp.used_features[idx] for idx in np.argsort(np.absolute(LIPEx_exp.local_exp[pred_label]))[::-1][:TopK]] # TopK features ranked descending
    LIPEx_TopK_words = [LIPEx_exp.domain_mapper.indexed_string.word(x) for x in LIPEx_TopK_features_idx]
    print("LIPEx_TopK_words:",LIPEx_TopK_words)

    TV_diatance = []
    for num in number_of_features_to_remove:
        f_repred_dis = f_re_pred_dis(idx,LIPEx_TopK_words[:num])
        LIPEx_repred_dis = exp_re_pred_dis(sample_data,LIPEx_TopK_features_idx[:num],LIPEx_exp,explainer.base.LIPEx_model)

        TV_diatance.append(TV(f_repred_dis,LIPEx_repred_dis))
    print('TV_diatance:',TV_diatance)
    remove_TV_distance.append(TV_diatance)
print('remove_TV_distance:',remove_TV_distance)

average_original_TV_distance = np.average(np.array(original_TV_distance),axis=0)
std_original_TV_distance = np.std(np.array(original_TV_distance),axis=0)

average_cross_TV_distance = np.average(np.array(remove_TV_distance),axis=0)
std_cross_TV_distance = np.std(np.array(remove_TV_distance),axis=0)

print('\n')
print('average_original_TV_distance:',average_original_TV_distance,'std_original_TV_distance:',std_original_TV_distance)
print('average_cross_TV_distance:',average_cross_TV_distance,'std_cross_TV_distance:',std_cross_TV_distance)