import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from datasets import load_dataset
from tqdm.notebook import tqdm
from sklearn.utils import Bunch
import random
random.seed(42)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_text import LimeTextExplainer


newsgroups = load_dataset('SetFit/20_newsgroups')
test_text, test_label = newsgroups ['test']['text'], newsgroups['test']['label']
new_test_text, new_text_label = [], []

for i in tqdm(range(len(test_text))):
    if len(set(test_text[i].split(' '))) > 30:
        new_test_text.append(test_text[i].lower())
        new_text_label.append(test_label[i])
assert len(new_test_text) == len(new_text_label)

new_test = Bunch(data=new_test_text, target=np.array(new_text_label))
print('Total texts in test dataset:',len(new_test.data)) 


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
PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Text/20NewsGroups/'
PRETRAINED_LM = "bert-base-uncased"

config = BertConfig.from_pretrained(PRETRAINED_LM)
config.num_labels = N_labels
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load(PATH+'model-30.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM)

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
    
    def predict(self, text, batch_size=16): # 32 for 3090, 128 for A100
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
    f_output = c.predict(processed_text) # from numpy to tensor
    _, f_predicted = torch.max(torch.tensor(f_output), 1)
    f_repred_label = f_predicted.detach().numpy()[0] #  Predicted top label
    return f_repred_label

def exp_re_pred(sample_data,explainer_TopK_features_idx,explainer_exp,explainer_model):
    # Re-Prediction of explainer
    input_instance_booling = sample_data[0].copy()
    input_instance_booling[explainer_TopK_features_idx] = 0
    new_instance_booling = torch.from_numpy(input_instance_booling[explainer_exp.used_features]).float()
    explainer_predict = explainer_model.predict(new_instance_booling.reshape(1,-1))
    _, explainer_predicted = torch.max(explainer_predict.data, 1)
    explainer_repred_label= explainer_predicted.detach().numpy()[0] #  Predicted top label
    return explainer_repred_label


TopK = 5 
number_of_features_to_remove = [1,2,3,4,5]
union_num = 3
num_of_test_documents = 100 # to be defined
random_test_documents_idx = random.sample(range(0,len(new_test.data)), num_of_test_documents)

LIPEx_count = []
for idx in random_test_documents_idx:
    LIPEx_count_list = []
    output = c.predict([new_test.data[idx]]) # (num_text, num_class)
    _, predicted = torch.max(torch.tensor(output), 1)
    pred_label= predicted.detach().numpy()[0] #  get the Predicted top label index

    print('Document ID:',idx,',','f_Predicted label:',label_names[pred_label],',','True label:',label_names[new_test.target[idx]]) 
    
    #------------------------Below for LIME and LIPEx ------------------------#
    explainer = LimeTextExplainer(class_names=label_names,random_state=42)
    # sample perturbation data, features2use: Union Set 
    sample_data, sample_labels, sample_distances, sample_weights, features2use = explainer.sample_data_and_features(new_test.data[idx], c.predict, num_features=union_num)

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
        LIPEx_repred = exp_re_pred(sample_data, LIPEx_TopK_features_idx[:num], LIPEx_exp, explainer.base.LIPEx_model)
        f_repred = f_re_pred(idx,LIPEx_TopK_words[:num])
        print('LIPEx_repred:',LIPEx_repred,',','f_repred:',f_repred)
        if f_repred == LIPEx_repred:
            LIPEx_count_list.append(1)
        else:
            LIPEx_count_list.append(0)
    LIPEx_count.append(LIPEx_count_list)
    print('--------End evaluation -----------')

average_LIPEx = np.average(np.array(LIPEx_count),axis=0)
print('average_LIPEx =',average_LIPEx.tolist())
