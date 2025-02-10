import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from datasets import load_dataset
from tqdm import tqdm
from sklearn.utils import Bunch
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_text import LimeTextExplainer

import time
start_time = time.time()


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
print('Total texts in test dataset:',len(new_test.data)) # 637

label_names = emotion["test"].features['label'].names
print('dataset_label:',label_names,type(label_names))
N_labels = len(label_names)

PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Text/Emotion'
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


def f_re_pred(idx,explainer_TopK_words):
   # Re-Prediction of predictor
    processed_text  = new_test.data[idx]
    for word in explainer_TopK_words:
        processed_text = processed_text.replace(word,'')
    f_output = c.predict([processed_text]) # from numpy to tensor
    _, f_predicted = torch.max(torch.tensor(f_output), 1)
    f_repred_label = f_predicted.detach().numpy()[0] #  Predicted top label
    return f_repred_label

def exp_re_pred(sample_data,explainer_TopK_features_idx,explainer_exp,explainer_model):
    # Re-Prediction of explainer
    new_sample_data = sample_data[0].copy()
    new_sample_data[explainer_TopK_features_idx] = 0
    new_sample_data = torch.from_numpy(new_sample_data[explainer_exp.used_features]).float()
    explainer_predict = explainer_model.predict(new_sample_data.reshape(1,-1))
    _, explainer_predicted = torch.max(explainer_predict.data, 1)
    explainer_repred_label= explainer_predicted.detach().numpy()[0] #  Predicted top label
    return explainer_repred_label

def random_chose_documents(dataset,num_of_documents):
    random_documents_idx = random.sample(range(0, len(dataset.data)), num_of_documents)
    return random_documents_idx

# -------------------START OF Re-prediction Comparison between LIPEx and LIME-------------------
TopK = 5 
number_of_features_to_remove = [1,2,3,4,5]

union_num = 5
num_samples = 1000 # to be defined
num_of_test_documents = 100 # to be defined

LIPEx_count = []
model_accuracy = 0
for idx in random_chose_documents(new_test,num_of_test_documents):
    LIPEx_count_list = []

    output = c.predict([new_test.data[idx]]) # (num_text, num_class)
    _, predicted = torch.max(torch.tensor(output), 1)
    pred_label= predicted.detach().numpy()[0] #  get the Predicted top label index
    
    if pred_label == new_test.target[idx]:
        model_accuracy += 1

    print('Document ID:',idx,',','f_Predicted label:',label_names[pred_label],',','True label:',label_names[new_test.target[idx]]) 
    
    #------------------------Below for LIME and LIPEx ------------------------#
    explainer = LimeTextExplainer(class_names=label_names,random_state=42)
    # sample perturbation data, features2use: Union Set 
    sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_test.data[idx], c.predict, num_features=union_num, num_samples=num_samples)

    # Compute LIPEx-List-s and LIME-List-s
    # needed: yss, sorted_labels?, data, distances, used_features, weights
    LIPEx_exp = explainer.explain_instance_LIPEx(
        sample_data,
        sample_labels,
        sample_distances,
        sample_weights,
        used_features=features2use,
        new_top_labels=N_labels
    )
    # LIPEx
    LIPEx_TopK_features_idx=[LIPEx_exp.used_features[idx] for idx in np.argsort(np.absolute(LIPEx_exp.local_exp[pred_label]))[::-1][:TopK]] # TopK features ranked descending
    LIPEx_TopK_words = [LIPEx_exp.domain_mapper.indexed_string.word(x) for x in LIPEx_TopK_features_idx]
    print("LIPEx_TopK_words:",LIPEx_TopK_words)

    print('--------Start evaluation -----------')
    for num in number_of_features_to_remove:
        LIPEx_repred_label = exp_re_pred(sample_data,LIPEx_TopK_features_idx[:num],LIPEx_exp,explainer.base.LIPEx_model)
        f_repred_label = f_re_pred(idx,LIPEx_TopK_words[:num])
        print('LIPEx_repred_label:',label_names[LIPEx_repred_label],',','f_repred_label:',label_names[f_repred_label])
        if f_repred_label == LIPEx_repred_label:
            LIPEx_count_list.append(1)
        else:
            LIPEx_count_list.append(0)
    LIPEx_count.append(LIPEx_count_list)
    print('--------End evaluation -----------')

print('model_accuracy:',model_accuracy/num_of_test_documents)

tracking_rate = np.average(np.array(LIPEx_count),axis=0)
print('tracking_rate =',tracking_rate.tolist())

end_time = time.time()
print('Time:',end_time-start_time)


# LIPEx_list = []
# for i in range(3):
#     LIPEx_count = []
#     for idx in random_chose_documents(new_test,num_of_test_documents):
#         LIPEx_count_list = []
#         output = c.predict([new_test.data[idx]]) # (num_text, num_class)
#         _, predicted = torch.max(torch.tensor(output), 1)
#         pred_label= predicted.detach().numpy()[0] #  get the Predicted top label index

#         print('Document ID:',idx,',','f_Predicted label:',label_names[pred_label],',','True label:',label_names[new_test.target[idx]]) 
        
#         #------------------------Below for LIME and LIPEx ------------------------#
#         explainer = LimeTextExplainer(class_names=label_names,random_state=42)
#         # sample perturbation data, features2use: Union Set 
#         sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_test.data[idx], c.predict, num_features=union_num)

#         # Compute LIPEx-List-s and LIME-List-s
#         # needed: yss, sorted_labels?, data, distances, used_features, weights
#         LIPEx_exp = explainer.explain_instance_LIPEx(
#             sample_data,
#             sample_labels,
#             sample_distances,
#             sample_weights,
#             used_features=features2use,
#             new_top_labels=N_labels
#         )
#         # LIPEx
#         LIPEx_TopK_features_idx=[LIPEx_exp.used_features[idx] for idx in np.argsort(np.absolute(LIPEx_exp.local_exp[pred_label]))[::-1][:TopK]] # TopK features ranked descending
#         LIPEx_TopK_words = [LIPEx_exp.domain_mapper.indexed_string.word(x) for x in LIPEx_TopK_features_idx]
#         print("LIPEx_TopK_words:",LIPEx_TopK_words)

#         print('--------Start evaluation -----------')
#         for num in number_of_features_to_remove:
#             LIPEx_repred_label = exp_re_pred(sample_data,LIPEx_TopK_features_idx[:num],LIPEx_exp,explainer.base.LIPEx_model)
#             f_repred_label = f_re_pred(idx,LIPEx_TopK_words[:num])
#             print('LIPEx_repred_label:',label_names[LIPEx_repred_label],',','f_repred_label:',label_names[f_repred_label])
#             if f_repred_label == LIPEx_repred_label:
#                 LIPEx_count_list.append(1)
#             else:
#                 LIPEx_count_list.append(0)
#         LIPEx_count.append(LIPEx_count_list)
#         print('--------End evaluation -----------')

#     LIPEx_list.append(np.average(np.array(LIPEx_count),axis=0).tolist())

# LIPEx_array = np.array(LIPEx_list)

# average_LIPEx = np.average(LIPEx_array,axis=0)
# std_LIPEx = np.std(LIPEx_array,axis=0)

# print('average_LIPEx =',average_LIPEx.tolist())
# print('std_LIPEx =',std_LIPEx.tolist())
