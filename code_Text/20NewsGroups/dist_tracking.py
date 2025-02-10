import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.datasets import fetch_20newsgroups

from sklearn.utils import Bunch
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# from lime_new.lime_text import LimeTextExplainer


OldtoNew = {0:1, 1:8, 2:9, 3:5, 4:14, 5:10, 6:17, 7:13, 8:7, 9:0, 10:19, 11:15, 12:6, 13:3, 14:16, 15:18, 16:12, 17:11, 18:2, 19:4}

newsgroups_test = fetch_20newsgroups(subset='test') # 7532
new_test_data, new_test_target = [], []


for i in range(len(newsgroups_test.data)):
    if len(set(newsgroups_test.data[i].split(' '))) > 30:
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

# Load model and tokenizer
PRETRAINED_LM = "paragsmhatre/20_news_group_classifier"
model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_LM)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_LM)

def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=512, padding=True, return_attention_mask=True, truncation=True, return_tensors='pt')

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Wrap tokenizer and model for LIME
class pipeline(object):
    def __init__(self, model, encoder): 
        self.model = model.to(device)
        self.encoder = encoder
        self.model.eval()
    
    def predict(self, text, batch_size=32): # 32 for 3090, 128 for A100
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
    processed_text  = new_test.data[idx]
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

cross_TV_distance = []
original_TV_distance = [] 

for idx in random_chose_documents(new_test,num_of_test_documents):

    f_output = c.predict([new_test.data[idx]]) # (num_text, num_class)
    _, predicted = torch.max(torch.tensor(f_output), 1)
    pred_label= predicted.detach().numpy()[0] #  get the Predicted top label index

    print('Document ID:',idx,',','f_Predicted label:',label_names[pred_label],',','True label:',label_names[new_test.target[idx]]) 
    
    #------------------------Below for  LIPEx ------------------------#
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
    cross_TV_distance.append(TV_diatance)

average_original_TV_distance = np.average(np.array(original_TV_distance),axis=0).tolist()
std_original_TV_distance = np.std(np.array(original_TV_distance),axis=0).tolist()

average_cross_TV_distance = np.average(np.array(cross_TV_distance),axis=0).tolist()
std_cross_TV_distance = np.std(np.array(cross_TV_distance),axis=0).tolist()

print('\n')
print('average_original_TV_distance =',average_original_TV_distance,'std_original_TV_distance =',std_original_TV_distance)
print('average_cross_TV_distance =',average_cross_TV_distance,'std_cross_TV_distance =',std_cross_TV_distance)