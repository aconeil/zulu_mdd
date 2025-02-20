from datasets import load_dataset 
import csv
from transformers import pipeline
import torch
import librosa
import time

#Use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#specify the model to load
asr = pipeline("automatic-speech-recognition", device=0, model="aconeil/w2v-bert-2.0-nchlt_mdd")

#Load the test dataset 
test_dataset = load_dataset("aconeil/mdd_zu", split="test")

#Specify name of output file
output = open("com_model_transcriptions.csv", "w")
writer = csv.writer(output)
writer.writerow(["model_output", "target", "scores", "scores_inserts"])

#Loop through each recording in the dataset to get the models transcription and write to csv file
count = 0
total_time = 0
for rec in test_dataset:
    count +=1
    bt = time.time()
    #Test audio sampling rate is 44100, but model requires resampling to 16000
    audio_array = librosa.resample(rec["audio"]["array"], orig_sr=44100, target_sr=16000)
    #inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    #with torch.no_grad():
    #    logits = model(**inputs).logits
    #predicted_ids = torch.argmax(logits, dim=-1)
    #model_out = processor.decode(predicted_ids)[0]
    model_out = asr(audio_array)
    target = rec["transcription"]
    score = rec["scores"]
    scores_inserts = rec["scores_inserts"]
    writer.writerow([model_out, target, score, scores_inserts])
    et = time.time()
    total_time += et-bt
    #Provide time left estimates at beginning and every 500
    if count == 100 or count%500 == 0:
        print("Remaining time estimate:", (total_time/count)*(len(test_dataset)-count))

