'''
This fine-tuning script is adapted from the tutorial found here: https://huggingface.co/blog/fine-tune-w2v2-bert
'''
import re
import numpy as np
from huggingface_hub import login
from datasets import concatenate_datasets
import random
import pandas as pd
import torch
from evaluate import load

login()

from datasets import load_dataset, Audio

nchlt_train = load_dataset("aconeil/nchlt", split="train")
nchlt_test = load_dataset("aconeil/nchlt", split="test")

nchlt_train = nchlt_train.remove_columns(["gender", "age", "speaker_id", "duration", "md5sum", "pdp_score"])
nchlt_test = nchlt_test.remove_columns(["gender", "age", "speaker_id", "duration", "md5sum", "pdp_score"])

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    print(df)


chars_to_remove_regex = '[(\[s\])\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]'

def remove_special_characters(batch):
    # remove special characters
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()

    return batch

nchlt_train = nchlt_train.map(remove_special_characters)


def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = nchlt_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=nchlt_train.column_names)
vocab_test = nchlt_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=nchlt_test.column_names)


vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

# extract unique characters again
vocab_train = nchlt_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=nchlt_train.column_names)
vocab_test = nchlt_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=nchlt_test.column_names)
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

repo_name = "w2v-bert-2.0-nchlt"

tokenizer.push_to_hub(repo_name)

from transformers import SeamlessM4TFeatureExtractor

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

from transformers import Wav2Vec2BertProcessor

processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.push_to_hub(repo_name)


nchlt_train = nchlt_train.cast_column("audio", Audio(sampling_rate=16_000))
nchlt_test = nchlt_test.cast_column("audio", Audio(sampling_rate=16_000))

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])

    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

nchlt_train = nchlt_train.map(prepare_dataset, remove_columns=nchlt_train.column_names)
nchlt_test = nchlt_test.map(prepare_dataset, remove_columns=nchlt_test.column_names)

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load("wer")

cer_metric = load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

from transformers import Wav2Vec2BertForCTC

model = Wav2Vec2BertForCTC.from_pretrained(
    "facebook/w2v-bert-2.0",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    add_adapter=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir='/N/slate/aconeil/w2v-bert-2.0-nchlt',
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=10,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=600,
  eval_steps=300,
  logging_steps=300,
  learning_rate=5e-5,
  warmup_steps=500,
  save_total_limit=2,
  push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=nchlt_train,
    eval_dataset=nchlt_test,
    tokenizer=processor.feature_extractor,
)

#Option to resume from training
trainer.train()#resume_from_checkpoint='')

trainer.push_to_hub()

