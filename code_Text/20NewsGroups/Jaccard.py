import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from sklearn.utils import Bunch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.datasets import fetch_20newsgroups

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import math
import random
random.seed(42)


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
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=512, padding=True, truncation=True, return_attention_mask=True, return_tensors='pt')

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Wrap tokenizer and model for LIME
class pipeline(object):
    def __init__(self, model, encoder): 
        self.model = model.to(device)
        self.encoder = encoder
        self.model.eval()
    
    def predict(self, text, batch_size=32): # 32 for 3090, 256 for A100
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

# -------------------START OF Re-prediction Comparison between LIPEx and LIME-------------------
TopK = 6 
union_num = 3 # To be defined 
num_of_test_documents = 3
random_test_documents_idx = random.sample(range(0, len(new_test.data)), num_of_test_documents)

delta_candidates =[math.pi/16, math.pi/8, math.pi/4, math.pi/2]

Jaccard_LIME = np.empty((0,4), int)
Jaccard_LIPEx = np.empty((0,4), int)
Jaccard_LIPEx_vs_LIME = np.empty((0,4), int)

count = np.empty((0,4), int)

for idx in random_test_documents_idx:
    # print('Input instance:',idx,'\n',new_test.data[idx]) # print the text of the current article
    output = c.predict([new_test.data[idx]]) # (num_text, num_class)
    _, predicted = torch.max(torch.tensor(output), 1)
    pred_label= predicted.detach().numpy()[0] #  Predicted top label index

    print('Document ID:',idx,',','f_Predicted label:',label_names[pred_label],',','True label:',label_names[new_test.target[idx]]) 

    explainer = LimeTextExplainer(class_names=label_names,random_state=42)
    # sample perturbation data, features2use: Union Set 
    sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_test.data[idx], c.predict, num_features=union_num)

    # Compute LIPEx-List-s and LIME-List-s
    # needed: yss, sorted_labels?, data, distances, used_features, weights
    LIME_exp, LIPEx_exp = explainer.explain_instance_LIPEx_LIME(
        sample_data,
        sample_labels,
        sample_distances,
        sample_weights,
        used_features=features2use,
        new_top_labels=N_labels,
        true_label=[pred_label]
    )
    # LIME
    LIME_used_features = [x[0] for x in LIME_exp.local_exp[pred_label][:TopK]]
    LIME_used_words=[LIME_exp.domain_mapper.indexed_string.word(x) for x in LIME_used_features]
    print('LIME_used_words:',LIME_used_words)
    
    # LIPEx
    LIPEx_used_features=[LIPEx_exp.used_features[idx] for idx in np.argsort(np.absolute(LIPEx_exp.local_exp[pred_label]))[::-1][:TopK]] 
    LIPEx_used_words = [LIPEx_exp.domain_mapper.indexed_string.word(x) for x in LIPEx_used_features] #lables for feature name
    print("LIPEx_used_words:",LIPEx_used_words)

    # Compute delta-LIPEx-List-s and delta-LIME-List-s
    
    LIME = []
    LIPEx = []
    LIPEx_vs_LIME =[]
    count_documents = []

    for delta in delta_candidates:
        threshold = math.cos(delta)
        sel_indices = []
        for i in range(len(sample_data)):
            if 1 - sample_distances[i]/100 >= threshold:
                sel_indices.append(i)
        delta_data, delta_labels, delta_distances, delta_weights = sample_data[sel_indices], sample_labels[sel_indices], sample_distances[sel_indices], sample_weights[sel_indices]
        
        count_documents.append(len(delta_data))

        LIME_exp_delta, LIPEx_exp_delta = explainer.explain_instance_LIPEx_LIME(
        delta_data,
        delta_labels,
        delta_distances,
        delta_weights,
        used_features=features2use,
        true_label=[pred_label], 
        new_top_labels=N_labels
    )
        # LIME
        LIME_used_features_delta = [x[0] for x in LIME_exp_delta.local_exp[pred_label][:TopK]]
        LIME_used_words_delta=[LIME_exp_delta.domain_mapper.indexed_string.word(x) for x in LIME_used_features_delta]
        print('LIME_used_words_delta:',LIME_used_words_delta)
        
        # LIPEx
        LIPEx_used_features_delta=[LIPEx_exp_delta.used_features[idx] for idx in np.argsort(np.absolute(LIPEx_exp.local_exp[pred_label]))[::-1][:TopK]] 
        LIPEx_used_words_delta = [LIPEx_exp_delta.domain_mapper.indexed_string.word(x) for x in LIPEx_used_features_delta] #lables for feature name
        print("LIPEx_used_words_delta:",LIPEx_used_words_delta)

        
        LIME.append(len(set(LIME_used_words_delta).intersection(set(LIME_used_words)))/len(set(LIME_used_words_delta).union(set(LIME_used_words))))
        LIPEx.append(len(set(LIPEx_used_words_delta).intersection(set(LIPEx_used_words)))/len(set(LIPEx_used_words_delta).union(set(LIPEx_used_words))))
        LIPEx_vs_LIME.append(len(set(LIPEx_used_words_delta).intersection(set(LIME_used_words)))/len(set(LIPEx_used_words_delta).union(set(LIME_used_words))))
    
    count = np.append(count, np.array([count_documents]), axis=0)
    Jaccard_LIME = np.append(Jaccard_LIME, np.array([LIME]), axis=0)
    Jaccard_LIPEx = np.append(Jaccard_LIPEx, np.array([LIPEx]), axis=0)
    Jaccard_LIPEx_vs_LIME = np.append(Jaccard_LIPEx_vs_LIME, np.array([LIPEx_vs_LIME]), axis=0)

count = np.average(count, axis=0)

Jaccard_LIME_average = np.average(Jaccard_LIME, axis=0)
Jaccard_LIPEx_average = np.average(Jaccard_LIPEx, axis=0)
Jaccard_LIPEx_vs_LIME_average = np.average(Jaccard_LIPEx_vs_LIME, axis=0)

Jaccard_LIME_std = np.std(Jaccard_LIME, axis=0)
Jaccard_LIPEx_std = np.std(Jaccard_LIPEx, axis=0)
Jaccard_LIPEx_vs_LIME_std = np.std(Jaccard_LIPEx_vs_LIME, axis=0)

print('Jaccard_LIME_average =',Jaccard_LIME_average.tolist())
print('Jaccard_LIPEx_average =',Jaccard_LIPEx_average.tolist())
print('Jaccard_LIPEx_vs_LIME_average =',Jaccard_LIPEx_vs_LIME_average.tolist())

print('Jaccard_LIME_std =',Jaccard_LIME_std.tolist())
print('Jaccard_LIPEx_std =',Jaccard_LIPEx_std.tolist())
print('Jaccard_LIPEx_vs_LIME_std =',Jaccard_LIPEx_vs_LIME_std.tolist())

print('Average pertabation samples under delta restriction :',count.tolist())

# # -------------------Plot Errorbar------------------
# delta_candidates_radians = [str(round(delta, ndigits=3)) for delta in delta_candidates]
# plt.figure(figsize=(16,16))
# fig, ax = plt.subplots()

# ax.errorbar(delta_candidates_radians, Jaccard_LIME_average.tolist(), yerr=Jaccard_LIME_std.tolist(), label='Jaccard_LIME')
# ax.errorbar(delta_candidates_radians, Jaccard_LIPEx_average.tolist(), yerr=Jaccard_LIPEx_std.tolist(), label='Jaccard_LIPEx')
# ax.errorbar(delta_candidates_radians, Jaccard_LIPEx_vs_LIME_average.tolist(), yerr=Jaccard_LIPEx_vs_LIME_std.tolist(), label='Jaccard_LIPEx_vs_LIME')

# plt.xlabel('delta (radians)')
# plt.ylabel('Jaccard index')
# plt.title("Jaccard index Vs delta")
# plt.legend()
# PATH = 'LIPEx/code_Text/Plot/Jaccard_ErrorPlot_Emotion.png'
# plt.savefig(PATH)
# plt.close()
