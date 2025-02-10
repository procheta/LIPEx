import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.utils import Bunch
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_text import LimeTextExplainer


# ## Loading data
# We'll use the [emotion](https://huggingface.co/datasets/emotion) dataset from the Hugging Face Hub.
# Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.
# We'll use the `load_dataset` function from the `datasets` library to load the dataset.

emotion = load_dataset('emotion')
test_text, test_label = emotion['test']['text'], emotion['test']['label']
new_test_text, new_text_label = [], []

for i in tqdm(range(len(test_text))):
    if len(set(test_text[i].split(' '))) > 20:
        new_test_text.append(test_text[i].lower())
        new_text_label.append(test_label[i])
assert len(new_test_text) == len(new_text_label)

new_test = Bunch(data=new_test_text, target=np.array(new_text_label))
print('Total texts in test dataset:',len(new_test.data)) #165

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
    
    def predict(self, text, batch_size=256):
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

def rePredict(idx,TopK_words,number_of_features_to_remove, pred_label):
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
num_of_test_documents = 10

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
    sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_test.data[idx], c.predict, num_features=union_num, num_samples=num_samples)

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
    # LIME reprediction  
    LIME_count_list = rePredict(idx,LIME_TopK_words,number_of_features_to_remove, pred_label_k)
    print('LIME_count_list:',LIME_count_list)

    # LIPEx
    LIPEx_TopK_features_idx=[LIPEx_exp.used_features[idx] for idx in np.argsort(np.absolute(LIPEx_exp.local_exp[pred_label_k]))[::-1][:TopK]] # TopK features ranked descending
    LIPEx_TopK_words = [LIPEx_exp.domain_mapper.indexed_string.word(x) for x in LIPEx_TopK_features_idx]
    print("LIPEx_TopK_words:",LIPEx_TopK_words)
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
