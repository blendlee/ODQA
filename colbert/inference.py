import json
import torch.nn.functional as F
import logging
import sys
from model import *
from tokenizer import *
from typing import Callable, Dict, List, NoReturn, Tuple
import torch
import numpy as np
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


def main():
    epoch=10
    datasets = load_from_disk('/opt/ml/input/data/test_dataset')
    test_dataset = datasets["validation"].flatten_indices().to_pandas()

    MODEL_NAME = 'klue/bert-base'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ColbertModel.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(tokenizer.vocab_size + 2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    model.load_state_dict(torch.load(f'/opt/ml/input/code/colbert/best_model/colbert_epoch{epoch}.pth'))
    #model.load_state_dict(torch.load(f'/opt/ml/input/code/colbert/best_model/colbert.pth'))
    model.to(device)


    print('opening wiki passage...')
    with open('/opt/ml/input/data/wikipedia_documents.json', "r", encoding="utf-8") as f:
        wiki = json.load(f)
    context = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print('wiki loaded!!!')

    query= list(test_dataset['question'])
    mrc_ids =test_dataset['id']
    length = len(test_dataset)


    batched_p_embs = []
    with torch.no_grad():
        model.eval
        q_seqs_val = tokenize_colbert(query,tokenizer,corpus='query').to('cuda')
        q_emb = model.query(**q_seqs_val).to('cpu')
        print(q_emb.size())


        print('Start passage embedding......')
        p_embs=[]
        for step,p in enumerate(tqdm(context)):
            p = tokenize_colbert(p,tokenizer,corpus='doc').to('cuda')
            p_emb = model.doc(**p).to('cpu').numpy()
            p_embs.append(p_emb)
            if (step+1)%200 ==0:
                batched_p_embs.append(p_embs)
                p_embs=[]
        batched_p_embs.append(p_embs)
    
    #q_emb = torch.cat([q_emb1,q_emb2], dim=1)


    dot_prod_scores = model.get_score(q_emb,batched_p_embs,eval=True)
    print(dot_prod_scores.size())

    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
    print(dot_prod_scores)
    print(rank)
    torch.save(rank,f'/opt/ml/input/code/inferecne_colbert_rank_epoch{epoch}.pth')
    print(rank.size())
    return 


if __name__ == '__main__':
    main()