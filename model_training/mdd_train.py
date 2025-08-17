import common

#Replace with desired HF repo name
repo_name = "w2v-bert-2.0-zuluMDD"

#Replace with desired output directory
output_dir = "DIR_NAME"

zulumdd_train = load_dataset("aconeil/zuluMDD", split="train")
zulumdd_train = zulumdd_train.remove_columns(["gender", "age", "scores", "scores_tones", "scores_inserts", "speaker", "l1","other_languages","semesters_study","pre_uni_years", "residency","birthplace"])

zulumdd_train = zulumdd_train.map(common.remove_special_characters)

vocab_train = zulumdd_train.map(common.extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=zulumdd_train.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(common.vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

common.tokenizer.push_to_hub(repo_name)

common.processor.push_to_hub(repo_name)

zulumdd_train = zulumdd_train.cast_column("audio", Audio(sampling_rate=16_000))

zulumdd_train = com_train.map(prepare_dataset, remove_columns=zulumdd_train.column_names)


model = Wav2Vec2BertForCTC.from_pretrained(
    common.base_model,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    add_adapter=True,
    pad_token_id=common.processor.tokenizer.pad_token_id,
    vocab_size=len(common.processor.tokenizer),
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
    data_collator=common.data_collator,
    args=training_args,
    compute_metrics=common.compute_metrics,
    train_dataset=zulumdd_train,
    eval_dataset=common.nchlt_test,
    tokenizer=common.processor.feature_extractor,
)

#Option to resume from checkpoint here
trainer.train()#resume_from_checkpoint='')

trainer.push_to_hub()
