
# Wav2Vec2-Bert for Mispronunciation Detection in isiZulu L2 speech

This repository contains the code used in the paper *title will be posted following anonymous review phase*.  

The folder ``model_training`` includes the scripts to fine-tune Wav2Vec2-Bert on the [NCHLT isiZulu corpus](https://repo.sadilar.org/handle/20.500.12185/275) (clean split) and the [isiZulu L2 Speech Corpus](https://repo.sadilar.org/handle/20.500.12185/685) teacher training and test on the NCHLT corpus. Both datasets are also now available on HuggingFace: [NCHLT isiZulu Corpus](https://huggingface.co/datasets/aconeil/nchlt) and [isiZulu L2 Speech Corpus](https://huggingface.co/datasets/aconeil/zuluMDD).  

``gen_transcripts.py`` uses the fine-tuned models to generate a transcription of each of the student recordings in the isiZulu L2 Speech Corpus and outputs them to a csv file. Note: this script takes a long time to run.

``group_results.py`` includes the functions for analysis used in the paper. These functions group the results of each fine-tuned model based on the models alignment to feedback of the strictest teacher. Using the groupings, various graphs are generated to understand the errors from each model. Additionally, print out statements for false acceptance rate, false rejecction rate, and true negative rate can be generated using this script.

The original database from the isiZulu L2 Speech Corpus ``zulu_mdd.db`` is included in the repo for validation of scores in the event that scores in one of the files has erroneously been converted to scientific notation with leading zeros dropped. If this is a concern, ``get_scores.py`` may be used to retrieve the scores from the database using filenames. 




