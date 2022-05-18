from datasets import DatasetDict, load_from_disk, load_metric
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import os
import pandas as pd
import torch
import torch.nn.functional as F
from dense_retrieval_package import *
import json
import pickle
from transformers import (
AutoConfig,
AutoModelForQuestionAnswering,
AutoTokenizer,
DataCollatorWithPadding,
EvalPrediction,
HfArgumentParser,
TrainingArguments,
set_seed,)


set_seed(42)
num_neg='in_batch'
args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01
)
#MODEL_NAME = 'bert-base-multilingual-cased'
MODEL_NAME = 'klue/bert-base'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_data('/opt/ml/input/data/train.csv')
#additional_dataset = load_data('/opt/ml/input/data/squad.csv')
#60407+3952
#additional_dataset = load_data('/opt/ml/input/data/qp_pair.csv')
additional_dataset = load_data('/opt/ml/input/data/masked_query_and_titled_passage.csv')
print('csv loading.....')
#additional_dataset squad
#train_dataset = dataset[:3952].append(additional_dataset[:-3])
#additional_dataset aug
train_dataset = dataset[:3952].append(additional_dataset)
#train_dataset = dataset[:3952]
train_dataset = train_dataset.reset_index(drop=True)
val_dataset = dataset[3952:]
print('dataset tokenizing.......')
train_context, train_query, train_title = tokenized_data(train_dataset, tokenizer,train=True, num_neg=num_neg)





tran_dataset = TensorDataset(train_context['input_ids'], train_context['attention_mask'], train_context['token_type_ids'], 
                        train_query['input_ids'], train_query['attention_mask'], train_query['token_type_ids'])


MODEL_NAME = 'klue/bert-base'
p_encoder = Dense_Retrieval_Model.from_pretrained(MODEL_NAME)
q_encoder = Dense_Retrieval_Model.from_pretrained(MODEL_NAME)








###############colbert####################


import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import AutoModel, BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup



class ColbertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(ColbertModel, self).__init__(config)

        self.similarity_metric = 'cosine'
        self.dim = 128
        self.batch = 8
        self.bert = BertModel(config)
        self.init_weights()
        self.linear = nn.Linear(config.hidden_size, self.dim, bias=False)  


    def forward(self, p_inputs,q_inputs):
        Q = self.query(**q_inputs)
        D = self.doc(**p_inputs)
        return self.get_score(Q,D)


    def query(self, input_ids, attention_mask, token_type_ids):
        Q = self.bert(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)


    def doc(self, input_ids, attention_mask, token_type_ids):
        D = self.bert(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)[0]
        D = self.linear(D)
        return torch.nn.functional.normalize(D, p=2, dim=2)

    def get_score(self,Q,D,eval=False):
        if eval:
            if self.similarity_metric == 'cosine':
                final_score=torch.tensor([])
                for D_batch in tqdm(D):
                    D_batch = torch.Tensor(D_batch).squeeze()
                    p_seqeunce_output=D_batch.transpose(1,2) #(batch_size,hidden_size,p_sequence_length)
                    q_sequence_output=Q.view(600,1,-1,self.dim) #(batch_size, 1, q_sequence_length, hidden_size)
                    dot_prod = torch.matmul(q_sequence_output,p_seqeunce_output) #(batch_size,batch_size, q_sequence_length, p_seqence_length)
                    max_dot_prod_score =torch.max(dot_prod, dim=3)[0] #(batch_size,batch_size,q_sequnce_length)
                    score = torch.sum(max_dot_prod_score,dim=2)#(batch_size,batch_size)
                    final_score = torch.cat([final_score,score],dim=1)
                print(final_score.size())
                return final_score
