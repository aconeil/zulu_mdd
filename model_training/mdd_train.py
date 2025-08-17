'''
This fine-tuning script is adapted from the tutorial found here: https://huggingface.co/blog/fine-tune-w2v2-bert
'''
import re
import numpy as np
from huggingface_hub import login
import random
import pandas as pd
import torch
from evaluate import load
from datasets import load_dataset, Audio
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2BertProcessor, SeamlessM4TFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2BertForCTC, TrainingArguments, Trainer

#Replace with desired HF repo name
repo_name = "w2v-bert-2.0-zuluMDD"

#Replace with desired output directory
output_dir = "DIR_NAME"

base_model = "facebook/w2v-bert-2.0"

chars_to_remove_regex = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\»\«]"

def remove_special_characters(batch):
    # remove special characters
    batch["transcription"] = re.sub(chars_to_remove_regex, "", batch["transcription"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

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
            return_tensors="pt",)
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",)
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

login()

zulumdd_train = load_dataset("aconeil/zuluMDD", split="train")
zulumdd_train = zulumdd_train.remove_columns(["gender", "age", "scores", "scores_tones", "scores_inserts", "speaker", "l1","other_languages","semesters_study","pre_uni_years", "residency","birthplace"])

nchlt_test = load_dataset("aconeil/nchlt", split="test")
nchlt_test = nchlt_test.remove_columns(["gender", "age", "speaker_id", "duration", "md5sum", "pdp_score"])


zulumdd_train = zulumdd_train.map(remove_special_characters)

vocab_train = zulumdd_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=zulumdd_train.column_names)

vocab_test = nchlt_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=nchlt_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

tokenizer.push_to_hub(repo_name)

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(base_model)

processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.push_to_hub(repo_name)

zulumdd_train = zulumdd_train.cast_column("audio", Audio(sampling_rate=16_000))
nchlt_test = nchlt_test.cast_column("audio", Audio(sampling_rate=16_000))

zulumdd_train = com_train.map(prepare_dataset, remove_columns=zulumdd_train.column_names)
nchlt_test = nchlt_test.map(prepare_dataset, remove_columns=nchlt_test.column_names)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load("wer")

cer_metric = load("cer")

model = Wav2Vec2BertForCTC.from_pretrained(
    base_model,
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

training_args = TrainingArguments(
  output_dir=output_dir,
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

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=zulumdd_train,
    eval_dataset=nchlt_test,
    tokenizer=processor.feature_extractor,
)

#Option to resume from checkpoint here
trainer.train()#resume_from_checkpoint='')

trainer.push_to_hub()
