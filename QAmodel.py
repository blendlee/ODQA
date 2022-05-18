
import pandas as pd
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaModel
from transformers import  BertModel, BertPreTrainedModel, AdamW,  TrainingArguments, get_linear_schedule_with_warmup
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class QAModel(BertPreTrainedModel):

    def __init__(self, config):
        super(QAModel, self).__init__(config)

        self.dim = 128
        self.bert = BertModel(config)
        self.init_weights()
        self.hidden_dim=768
        self.batch_size=32
        #self.loss_fct=nn.CrossEntropyLoss()

        self.start_lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                           batch_first= True, bidirectional= True)
        self.end_lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                           batch_first= True, bidirectional= True)

        self.start_qlinear = nn.Linear(config.hidden_size*2, self.dim, bias=False)  
        self.start_dlinear = nn.Linear(config.hidden_size, self.dim, bias=False)  
        self.end_qlinear = nn.Linear(config.hidden_size*2, self.dim, bias=False)  
        self.end_dlinear = nn.Linear(config.hidden_size, self.dim, bias=False)  

    def masking(self,outputs,mask):
        Q= torch.masked_select(outputs,(mask.unsqueeze(-1)==1))
        #D= torch.masked_select(outputs,(mask.unsqueeze(-1)==0))
        Q=Q.view([-1,30,self.hidden_dim]) #batch,seq_len,dim
        D=outputs.view([-1,256,self.hidden_dim])
        return Q,D

    def forward(self,input_ids,attention_mask,token_type_ids,mask,start_positions=None,end_positions=None,return_dict = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=return_dict)
        #mask=torch.tensor(mask)
        Q,D = self.masking(outputs[0],mask)

        start_logits = self.start_embedding(Q,D)
        end_logits = self.end_embedding(Q,D)

        total_loss=None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def start_embedding(self,Q,D):
        _, (last_hidden, _)= self.start_lstm(Q)
        Q= torch.cat((last_hidden[0], last_hidden[1]), dim= 1) #(batch_size, hidden_dim*2)
        Q = self.start_qlinear(Q)
        Q= torch.nn.functional.normalize(Q, p=2, dim=1) #(direction_num,hidden_size)
        D = self.start_dlinear(D)
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        Q=Q.unsqueeze(-1)

        start_logits=F.softmax(torch.bmm(D,Q).squeeze(2),dim=1)
        #zero = torch.zeros([self.batch_size,30])

        return start_logits

    def end_embedding(self,Q,D):
        _, (last_hidden, _)= self.end_lstm(Q)
        Q= torch.cat((last_hidden[0], last_hidden[1]), dim= 1) #(batch_size, hidden_dim*2)
        Q = self.end_qlinear(Q)
        Q= torch.nn.functional.normalize(Q, p=2, dim=1) #(direction_num,hidden_size)
        D = self.end_dlinear(D)
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        Q=Q.unsqueeze(-1)
        #D=D.transpose(1,2)
        end_logits=F.softmax(torch.bmm(D,Q).squeeze(2),dim=1)
        return end_logits

class QAModelTEST(BertPreTrainedModel):

    def __init__(self, config):
        super(QAModelTEST, self).__init__(config)

        self.dim = 128
        self.bert = BertModel(config)
        self.init_weights()
        self.hidden_dim=768
        self.lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 1, dropout= 0.2,
                           batch_first= True, bidirectional= False)
        self.linear = nn.Linear(config.hidden_size, 2, bias=False) 
        #self.linear = nn.Linear(1, 2, bias=False) 

    def forward(self,input_ids,attention_mask,token_type_ids,
                    q_input_ids, q_attention_mask, q_token_type_ids,
                    start_positions=None,end_positions=None,return_dict = None):
        
        query_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=return_dict)
        context_outputs = self.bert(input_ids=q_input_ids, attention_mask=q_attention_mask, token_type_ids=q_token_type_ids,return_dict=None)
        

        output,(hn,cn) = self.lstm(query_outputs[0])
        query_sequence = output
        #print(hn.squeeze(0).unsqueeze(2).size())
        #similarity = torch.matmul(context_outputs[0],hn.squeeze(0).unsqueeze(2))
        #logits=self.linear(similarity)

        similarity = query_sequence*context_outputs[0]
        logits=self.linear(similarity)
        

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss=None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + query_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=query_outputs.hidden_states,
            attentions=query_outputs.attentions,
        )

class QAModelBASE(BertPreTrainedModel):

    def __init__(self, config):
        super(QAModelBASE, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

        #self.lstm =nn.LSTM(input_size= config.hidden_size, hidden_size= config.hidden_size, num_layers= 2, dropout= 0.2,
        #                   batch_first= True, bidirectional= True)
        self.linear = nn.Linear(config.hidden_size, 2, bias=False)  

    def forward(self,input_ids,attention_mask,token_type_ids,start_positions=None,end_positions=None,return_dict = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=return_dict)

        #output,(hn,cn)=self.lstm(outputs[0])
        logits=self.linear(outputs[0])

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss=None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class QAModelBASE(BertPreTrainedModel):

    def __init__(self, config):
        super(QAModelBASE, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

        #self.lstm =nn.LSTM(input_size= config.hidden_size, hidden_size= config.hidden_size, num_layers= 2, dropout= 0.2,
        #                   batch_first= True, bidirectional= True)
        self.linear = nn.Linear(config.hidden_size, 2, bias=False)  

    def forward(self,input_ids,attention_mask,token_type_ids,start_positions=None,end_positions=None,return_dict = None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=return_dict)

        #output,(hn,cn)=self.lstm(outputs[0])
        logits=self.linear(outputs[0])

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss=None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




class RobertaQA(RobertaPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super(RobertaQA,self).__init__(config)


        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.lstm = nn.LSTM(input_size= config.hidden_size, hidden_size= config.hidden_size, num_layers= 2, dropout= 0.2,
                              batch_first= True, bidirectional= False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        #sequence_output,(hn,cn) = self.lstm(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


