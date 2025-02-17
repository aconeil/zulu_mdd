from datasets import load_dataset 
import csv
from transformers import AutoProcessor, Wav2Vec2BertForCTC
import torch
import librosa

#specify the model to load
processor = AutoProcessor.from_pretrained("aconeil/w2v-bert-2.0-nchlt")
model = Wav2Vec2BertForCTC.from_pretrained("aconeil/w2v-bert-2.0-nchlt") 

#Load the test dataset 
test_dataset = load_dataset("aconeil/zuluMDD", split="test")

#Specify name of output file
output = open("nchlt_model_transcriptions.csv", "w")
writer = csv.writer(output)
writer.writerow(["model_output", "target", "scores", "scores_inserts"])

#Loop through each recording in the dataset to get the models transcription and write to csv file
#Note: This is likely to take a few hours
for rec in test_dataset:
    #Test audio sampling rate is 44100, but model requires resampling to 16000
    audio_array = librosa.resample(rec["audio"]["array"], orig_sr=44100, target_sr=16000)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    model_out = processor.batch_decode(predicted_ids)[0]
    target = rec["transcription"]
    score = rec["scores"]
    scores_inserts = rec["scores_inserts"]
    writer.writerow([model_out, target, score, scores_inserts])

