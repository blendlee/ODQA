import logging
import os
import sys
from typing import NoReturn

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric, load_dataset,concatenate_datasets
from trainer_qa import QuestionAnsweringTrainer
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
from utils_qa import check_no_error, postprocess_qa_predictions
parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
global data_args
model_args, data_args, training_args = parser.parse_args_into_dataclasses()




def prepare_train_features(examples,tokenzier):

    tokenized_document = tokenzier(
            examples['context'],
            truncation=True,
            max_length=512,
            stride=256,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True
        )

    sample_mapping = tokenized_document.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_document.pop("offset_mapping")

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_document["input_ids"][i]
        cls_index = 0

        sequence_ids = tokenized_document.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples['answer'][sample_index]

        #context안의 answer_start index와 answer_end_index
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        #answer = answers['text'][0]
        #tokenized_answer = tokenizer.encode(answer)[1:-1]


        # text에서 current span의 Start token index
        token_start_index = 0
        while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
            token_start_index += 1

        # text에서 current span의 End token index
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
            token_end_index -= 1


        if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
                ):
            tokenized_document["start_positions"].append(cls_index)
            tokenized_document["end_positions"].append(cls_index)
        else:
            while(
                token_start_index < len(offsets) 
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            tokenized_document["start_positions"].append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_document["end_positions"].append(token_end_index + 1)

    return tokenized_document
