import torch
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForCausalLM
import random

import sys
sys.path.append("/content/drive/MyDrive/LIPEx-704A/lime_new")
from lime_text import LimeTextExplainer

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# from lime_new.lime_text import LimeTextExplainer
import pandas as pd
import torch
import pickle
import numpy as np
from transformers import AutoModelForCausalLM,AutoTokenizer

OldtoNew = {0:1, 1:8, 2:9, 3:5, 4:14, 5:10, 6:17, 7:13, 8:7, 9:0, 10:19, 11:15, 12:6, 13:3, 14:16, 15:18, 16:12, 17:11, 18:2, 19:4}

newsgroups_test = fetch_20newsgroups(subset='test') # 7532
# Filter test data, only keep the documents with more than 100 unique words
new_test_data, new_test_target = [], []
for i in range(len(newsgroups_test.data)):
    if len(set(newsgroups_test.data[i].split(' '))) > 100:
        new_test_data.append(newsgroups_test.data[i].lower())
        new_test_target.append(OldtoNew[newsgroups_test.target[i]])
assert len(new_test_data) == len(new_test_target)

new_test = Bunch(data=new_test_data, target=np.array(new_test_target))
print('Total texts in test dataset:',len(new_test.data)) # 4673

label_names = ["rec.sport.baseball", 
              "alt.atheism",
              "talk.politics.misc",
              "sci.med",
              "talk.religion.misc",
              "comp.sys.ibm.pc.hardware",
              "sci.electronics",
              "rec.motorcycles",
              "comp.graphics",
              "comp.os.ms-windows.misc",
              "comp.windows.x",
              "talk.politics.mideast",
              "talk.politics.guns",
              "rec.autos",
              "comp.sys.mac.hardware",
              "sci.crypt",
              "sci.space",
              "misc.forsale",
              "soc.religion.christian",
              "rec.sport.hockey"]

print('dataset_label:',label_names,type(label_names))
N_labels = len(label_names)

# # Load model and tokenizer
# PRETRAINED_LM = "paragsmhatre/20_news_group_classifier"
PRETRAINED_LM = "openai-community/gpt2-large"
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_LM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_LM)

tokenizer.pad_token = tokenizer.eos_token

def encode(docs):
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=512, padding=True, truncation=True, return_attention_mask=True, return_tensors='pt')
    # encoded_dict = tokenizer(
    #     docs, 
    #     add_special_tokens=True, 
    #     max_length=512, 
    #     padding=True, 
    #     truncation=True, 
    #     return_tensors='pt'
    # )
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Wrap tokenizer and model for LIME
class pipeline(object):
    def __init__(self, model, encoder): 
        self.model = model.to(device)
        self.encoder = encoder
        self.model.eval()

    def predict(self, text, batch_size = 32): # batch predict mainly for LIPEx sample data prediction
        num_batches = int(len(text)/batch_size) if len(text)%batch_size == 0 else int(len(text)/batch_size)+1
        out = []
        for num in range(num_batches):
            batch_text = text[num*batch_size:(num+1)*batch_size]

            batch_input_ids, batch_attention_mask = self.encoder(batch_text)

            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)

            batch_output = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits # (batch_size, num_class)
            batch_out = batch_output.softmax(dim=-1).cpu().detach().tolist() # (batch_size, num_class)
            out += batch_out
        return np.array(out)
        

c = pipeline(model,encoder=encode)

def rePredict(idx, TopK_words, number_of_features_to_remove, pred_label):
    count_list = []
    for num in number_of_features_to_remove:
        count_change = 0
        copy_text = new_test.data[idx]
        for i in range(num):
            word = TopK_words[i]
            copy_text = copy_text.replace(word,'')
        output = c.predict(copy_text) # from numpy to tensor
        predicted = torch.argsort(torch.tensor(output), 1, descending = True)
        repred_label_k= predicted[0].detach().tolist()[1] # second Predicted top label index
        if repred_label_k != pred_label:
            count_change = 1
        count_list.append(count_change)
    return count_list

def random_test_documents(num_of_test_documents):
    random_test_documents_idx = random.sample(range(0, len(new_test.data)), num_of_test_documents)
    return random_test_documents_idx

TopK = 5
union_num = 5
number_of_features_to_remove = [1,2,3,4,5]
num_samples=1000
num_of_test_documents = 3


LIME_count, LIPEx_count = [], []
for idx in random_test_documents(num_of_test_documents):
    # print('Input instance:',idx,'\n',new_test.data[idx]) # print the text of the current article
    output = c.predict([new_test.data[idx]]) # (num_text, num_class)
    predicted = torch.argsort(torch.tensor(output), 1, descending = True)
    pred_label_k= predicted[0].detach().tolist()[1] # second Predicted top label index
    print('Predicted label:',label_names[pred_label_k])

    #------------------------Below for LIME and LIPEx ------------------------#
    explainer = LimeTextExplainer(class_names=label_names,random_state=42)
    # sample perturbation data, features2use: Union Set 
    sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_test.data[idx], c.predict, num_features = union_num, num_samples = num_samples)

    # Compute LIPEx-List-s and LIME-List-s
    # needed: yss, sorted_labels?, data, distances, used_features, weights
    LIME_exp, LIPEx_exp = explainer.explain_instance_LIPEx_LIME(
        sample_data,
        sample_labels,
        sample_distances,
        sample_weights,
        used_features=features2use,
        new_top_labels=N_labels,
        true_label=[pred_label_k]
    )

    # LIME
    LIME_TopK_features_idx = [x[0] for x in LIME_exp.local_exp[pred_label_k][:TopK]] # TopK features ranked descending
    LIME_TopK_words=[LIME_exp.domain_mapper.indexed_string.word(x) for x in LIME_TopK_features_idx]
    print('LIME_TopK_words:',LIME_TopK_words)
    if len(LIME_TopK_words) < TopK:
        print ('LIME_TopK_words is less than',TopK)
        continue
    # LIME reprediction  
    LIME_count_list = rePredict(idx,LIME_TopK_words,number_of_features_to_remove, pred_label_k)
    print('LIME_count_list:',LIME_count_list)

    # LIPEx
    LIPEx_TopK_features_idx=[LIPEx_exp.used_features[idx] for idx in np.argsort(np.absolute(LIPEx_exp.local_exp[pred_label_k]))[::-1][:TopK]] # TopK features ranked descending
    LIPEx_TopK_words = [LIPEx_exp.domain_mapper.indexed_string.word(x) for x in LIPEx_TopK_features_idx]
    print("LIPEx_TopK_words:",LIPEx_TopK_words)
    if len(LIPEx_TopK_words) < TopK:
        print ('LIPEx_TopK_words is less than',TopK)
        continue
    # LIPEx reprediction 
    LIPEx_count_list = rePredict(idx,LIPEx_TopK_words,number_of_features_to_remove, pred_label_k)
    print('LIPEx_count_list:',LIPEx_count_list)

    #------------------------End------------------------#
    LIME_count.append(LIME_count_list)
    LIPEx_count.append(LIPEx_count_list)
    assert len(LIPEx_count) == len(LIME_count)

LIME_array = np.array(LIME_count)
LIPEx_array = np.array(LIPEx_count)

average_LIME = LIME_array.sum(axis=0) / num_of_test_documents
average_LIPEx = LIPEx_array.sum(axis=0) / num_of_test_documents
print('LIME:',average_LIME)
print('LIPEx:',average_LIPEx)       
