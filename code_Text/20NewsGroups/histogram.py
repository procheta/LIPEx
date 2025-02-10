import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.utils import Bunch
import random

from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from sklearn.datasets import fetch_20newsgroups

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_text import LimeTextExplainer

# import time
# start_time = time.time()

newsgroups_test = fetch_20newsgroups(subset='test') # 7532
new_test_data, new_test_target = [], []
# print('Total texts in fetch_20newsgroups:',len(newsgroups_test.data)) 

for i in range(len(newsgroups_test.data)):
    if len(set(newsgroups_test.data[i].split(' '))) > 30:
        new_test_data.append(newsgroups_test.data[i].lower())
        new_test_target.append(newsgroups_test.target[i])
assert len(new_test_data) == len(new_test_target)

new_test = Bunch(data=new_test_data, target=np.array(new_test_target))
print('Total texts in test dataset:',len(new_test.data)) # 7365

label_names = ['alt.atheism',
'comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'comp.windows.x',
'misc.forsale',
'rec.autos',
'rec.motorcycles',
'rec.sport.baseball',
'rec.sport.hockey',
'sci.crypt',
'sci.electronics',
'sci.med',
'sci.space',
'soc.religion.christian',
'talk.politics.guns',
'talk.politics.mideast',
'talk.politics.misc',
'talk.religion.misc']

print('dataset_label:',label_names,type(label_names))
N_labels = len(label_names)

# Load model and tokenizer
PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Text/20NewsGroups/best_ckpt/'
PRETRAINED_LM = "bert-base-uncased"

config = BertConfig.from_pretrained(PRETRAINED_LM)
config.num_labels = N_labels
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load(PATH+'model-15.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)

def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=512, padding=True, return_attention_mask=True, truncation=True, return_tensors='pt') # max_length to be defined

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Wrap tokenizer and model for LIME
class pipeline(object):
    def __init__(self, model, encoder): 
        self.model = model.to(device)
        self.encoder = encoder
        self.model.eval()
    
    def predict(self, text, batch_size=16):  # 16 for 3090, 64 for A100
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

def random_chose_documents(dataset,num_of_documents):
    random_documents_idx = random.sample(range(0, len(dataset.data)), num_of_documents)
    return random_documents_idx

TopK = 5 
number_of_features_to_remove = [1,2,3,4,5]

union_num = 5
num_samples = 1000 # to be defined
num_of_test_documents = 3 # to be defined

TV_s = [] 

for idx in random_chose_documents(new_test,num_of_test_documents):
   
    f_output = c.predict([new_test.data[idx]]) # (num_text, num_class)
    _, predicted = torch.max(torch.tensor(f_output), 1)
    pred_label= predicted.detach().numpy()[0] #  get the Predicted top label index

    print('Document ID:',idx,',','f_Predicted label:',label_names[pred_label],',','True label:',label_names[new_test.target[idx]]) 
    
    #------------------------Below for LIME and LIPEx ------------------------#
    explainer = LimeTextExplainer(class_names=label_names,random_state=42)
    # sample perturbation data, features2use: Union Set 
    sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_test.data[idx], c.predict, num_features=union_num, num_samples=num_samples)

    LIPEx_exp = explainer.explain_instance_LIPEx(
        sample_data,
        sample_labels,
        sample_distances,
        sample_weights,
        used_features=features2use,
        new_top_labels=N_labels
    )
    print('TV_distance',TV(torch.from_numpy(f_output),LIPEx_exp.local_pred))
    TV_s.append(TV(torch.from_numpy(f_output),LIPEx_exp.local_pred))

print('\n')
print('TV(f(s), LIPEx(s)) =',TV_s)
print('len(TV_s)) =',len(TV_s))

print('Avg(TV(f(s), LIPEx(s))) =',np.mean(TV_s))
print('Std(TV(f(s), LIPEx(s))) =',np.std(TV_s))
   