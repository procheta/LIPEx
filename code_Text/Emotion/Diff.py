import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
# import transformers
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from datasets import load_dataset
from tqdm import tqdm
from sklearn.utils import Bunch
import copy

import random
random.seed(42)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from lime_new.lime_text import LimeTextExplainer

emotion = load_dataset('emotion')
test_text, test_label = emotion['test']['text'], emotion['test']['label']
new_test_text, new_text_label = [], []

for i in tqdm(range(len(test_text))):
    if len(set(test_text[i].split(' '))) > 30: 
        new_test_text.append(test_text[i].lower())
        new_text_label.append(test_label[i])
assert len(new_test_text) == len(new_text_label)

# when length > 30, total texts in test dataset: 165
# when length > 25, total texts in test dataset: 359

new_test = Bunch(data=new_test_text, target=np.array(new_text_label))
print('Total texts in test dataset:',len(new_test.data)) #165

label_names = emotion["test"].features['label'].names
print('dataset_label:',label_names,type(label_names))
N_labels = len(label_names)

PATH = '/mnt/b432dc15-0b9a-4f76-9076-5bf99fe91d74/Hongbo/LIPEx/code_Text/Emotion'
PRETRAINED_LM = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM, do_lower_case=True)

config = BertConfig.from_pretrained(PRETRAINED_LM)
config.num_labels = N_labels
model = BertForSequenceClassification(config)

model.load_state_dict(torch.load(PATH+'/model-25.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('GPU:',torch.cuda.get_device_name(device=device))

def gaussian_noise(x, var):
  return torch.normal(0, var, size=x.shape).to(x.device)

def add_noise(layer, var):
    with torch.no_grad():
        for _, param in layer.named_parameters():
                param.add_(gaussian_noise(param, var))
        return

def encode(docs):
    '''
    This function takes list of texts and returns input_ids and attention_mask of texts
    '''
    encoded_dict = tokenizer(docs, add_special_tokens=True, max_length=128, padding='max_length',truncation=True, return_attention_mask=True, return_tensors='pt')

    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    return input_ids, attention_masks

# Wrap tokenizer and model for LIME
class pipeline(object):
    def __init__(self, model, encoder): 
        self.model = model.to(device)
        self.encoder = encoder
        self.model.eval()
    
    def predict(self, text, batch_size=64):
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

def Accuracy(model, test_data):
    '''
    This function takes model and test_data and returns accuracy
    '''
    output = model.predict(test_data.data)
    pred_label = np.argmax(output,axis=-1).tolist()
    target_label = test_data.target.tolist()
    count = 0
    for i in range(len(pred_label)):
        if pred_label[i] == target_label[i]:
            count += 1
    return count/len(pred_label)

# distance metrics of two distributions
def TV(p, q):
        TotalVar_D = torch.mean(0.5 * (torch.sum(torch.abs(p-q), dim=-1))) 
        return TotalVar_D.item()

union_num = 5
num_of_test_documents = 10
random_test_documents_idx = random.sample(range(0, len(new_test.data)), num_of_test_documents)

accuracy_list = []
model_LIPEx_TV_average_list = []

var_list = [round(var,2) for var in np.arange(0, 1, 0.1).tolist()]

for var in var_list:
    distort_model = copy.deepcopy(model)
    add_noise(distort_model.classifier,var)
    distort_c = pipeline(distort_model,encoder=encode)
    accuracy_list.append(Accuracy(distort_c,new_test))

    model_LIPEx_TV_distance = []

    for idx in random_test_documents_idx:
        print('Test instance:', new_test.data[idx])
        explainer = LimeTextExplainer(class_names=label_names,random_state=42)
        
        #--------------------Below for distort model -----------------------------#
        distort_output = distort_c.predict([new_test.data[idx]]) # (num_text, num_class)
        
        distort_sample_data, distort_sample_labels, distort_sample_distances, distort_sample_weights, distort_features2use = explainer.sample_data_and_features(new_test.data[idx], distort_c.predict, num_features=union_num, num_samples=1000)

        distort_LIPEx_exp = explainer.explain_instance_LIPEx(
                distort_sample_data,
                distort_sample_labels,
                distort_sample_distances,
                distort_sample_weights,
                used_features=distort_features2use,
                new_top_labels=N_labels
            )
        distort_local_pred = distort_LIPEx_exp.local_pred
        
        # compute TV distance of two model output distributions
        model_LIPEx_TV_distance.append(TV(torch.from_numpy(distort_output), distort_local_pred))

    model_LIPEx_TV_distance = np.array(model_LIPEx_TV_distance)
    model_LIPEx_TV_average = np.mean(model_LIPEx_TV_distance, axis=0)

    model_LIPEx_TV_average_list.append(model_LIPEx_TV_average)
 
print('var_list =',var_list)
print('accuracy_list =',accuracy_list)
print('model_LIPEx_TV_average_list =', model_LIPEx_TV_average_list)
