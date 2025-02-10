import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.utils import Bunch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_text import LimeTextExplainer

import matplotlib.pyplot as plt
import math
import random
random.seed(42)


# ## Loading data
# We'll use the [emotion](https://huggingface.co/datasets/emotion) dataset from the Hugging Face Hub.
# The dataset contains 416809 tweets with 6 emotions: anger, joy, optimism, sadness, and neutral.
# We'll use the `load_dataset` function from the `datasets` library to load the dataset.

emotion = load_dataset('emotion')
test_text, test_label = emotion['test']['text'], emotion['test']['label']
new_test_text, new_text_label = [], []

for i in tqdm(range(len(test_text))):
    if len(set(test_text[i].split(' '))) > 30:
        new_test_text.append(test_text[i].lower())
        new_text_label.append(test_label[i])
assert len(new_test_text) == len(new_text_label)

new_test = Bunch(data=new_test_text, target=np.array(new_text_label))
print('Total texts in test dataset:',len(new_test.data)) # 165

label_names = emotion["test"].features['label'].names
print('dataset_label:',label_names,type(label_names))
N_labels = len(label_names)

PATH = 'LIPEx/code_Text/Emotion'
PRETRAINED_LM = "bert-base-uncased"

config = BertConfig.from_pretrained(PRETRAINED_LM)
config.num_labels = N_labels
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load(PATH+'/model-25.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM, do_lower_case=True)

def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=128, padding=True, truncation=True, return_attention_mask=True, return_tensors='pt')

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Wrap tokenizer and model for LIME
class pipeline(object):
    def __init__(self, model, encoder): 
        self.model = model.to(device)
        self.encoder = encoder
        self.model.eval()
    
    def predict(self, text, batch_size=256): #256 for 3090, 512 for A100
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
union_num = 5
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
