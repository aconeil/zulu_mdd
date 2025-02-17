'''
This script processes the output from the gen_transcripts.py script by making a folder that groups the results of each model
To run: python group_results.py transcript.csv output_folder_name
'''

import sys
import os
import pandas as pd
import string
import re
import numpy as np
import ast
import zulu2ipa as zp
from sequence_align.pairwise import needleman_wunsch
import matplotlib.pyplot as plt
from collections import Counter
from scipy.interpolate import make_interp_spline

#model_df = pd.read_csv(sys.argv[1], usecols=['Unnamed: 0', 'model_output', 'target', 'filename', 'db_score', 'insert'])[['Unnamed: 0', 'model_output', 'target', 'filename', 'db_score', 'insert']]
#out_folder = os.mkdir(sys.argv[2])

toIPA = {"1":"ǀ̬̃", "|":"\\s", "2":"ǃ̬̃", "3":"ǁ̬̃", "K":"kʰ", "P":"pʰ", "T":"tʰ", "C":"ǀʰ", "4":"ǀ̬", "7":"ǀ̃", "Q":"ǃʰ", "X":"ǁʰ", "6":"ǃ̬", "5":"ǁ̬", "9":"ǃ̃", "8":"ǁ̃", "W":"lw", "x":"ǁ", "q":"ǃ", "c":"ǀ"}

problem_clicks = ["4", "6", "9", "5"]

x_order = ["a", "e", "i", "o", "u", "p", "pʰ", "ɓ", "b", "t", "tʰ", "d", "k", "kʰ", "g", "m", "n", "ɲ", "ŋ", "f", "v", "s", "z", "ʃ", "ʤ", "h", "ɦ", "ɬ", "ɮ", "j", "l", "w", "ǀ", "ǀʰ", "ǀ̃", "ǀ̬", "ǀ̬̃", "ǃ", "ǃʰ", "ǃ̃","ǃ̬", "ǃ̬̃", "ǁ", "ǁʰ", "ǁ̃", "ǁ̬", "ǁ̬̃", "\\s", "EOS"]

#Function to lower case and remove all punctuation
def standardize(spelling):
    depunctuate = str.maketrans('', '', string.punctuation)
    spelling = spelling.translate(depunctuate)
    spelling = spelling.lower()
    return spelling

def fix_lw(text, score):
    score_index = text.index("W")
    rep_score = score[score_index]
    score = score[:score_index] + rep_score + score[score_index:]
    text = text.replace("W", "lw", 1)
    if "W" in text:
        score, text = fix_lw(text, score)
    return score, text
    
def fix_clicks(text, score, err):
    score_index = text.index(err)
    score = score[:score_index] + score[score_index+1:]
    return score

#Function that returns the index of inserts
def extract(inserts):
    inserts = inserts.replace("None", "'[]'")
    inserts = ast.literal_eval(inserts)
    max_inserts = max(inserts, key=len)
    max_inserts = ast.literal_eval(max_inserts)
    if max_inserts == []:
        return []
    else:
        numbers = [int(n) for n in max_inserts]
        return numbers

#Function that removes skipped feedback and converts to list
def process_scores(scores):
    no_skips = []
    scores = ast.literal_eval(scores)
    for score in scores:
        if score != 'SKIPPED':
            no_skips.append(score)
    return(no_skips)

#Function to check whether any of the teachers scored a phoneme as incorrect
def perfect_feedback(scores):
    for score in scores:
        #Zero corresponds to a mistake
        if "0" in score:
            return False
    return True

def preprocess(df):
    df['db_score'] = df['db_score'].apply(process_scores)
    df['target'] = df['target'].apply(standardize).apply(zp.zulu2ipa)
    df['model_output'] = df['model_output'].apply(standardize).apply(zp.zulu2ipa)
    df['insert'] = df['insert'].apply(extract)
    return df

def test_tie(scores):
    return all(x == scores[0] for x in scores)

def choose_score(scores):
    if scores == []:
        return False
    most_mistakes = 0
    pick_index = 0
    for i in range(len(scores)):
        mistakes = scores[i].count("0")
        if mistakes >= most_mistakes:
            most_mistakes = mistakes
            pick_index = i
    #TODO make it so that it intelligentally chooses between equally bad scores
    return scores[pick_index]

def error_indices(score):
    indices = []
    for i,c in enumerate(score):
        if c == "0":
            indices.append(i)
    return indices

def get_score(model, target):
    mod_score = ""
    mod_inserts = []
    for i in range(len(target)):
        if target[i] == model[i]:
            mod_score = mod_score + "1"
        elif target[i] == "-":
            try:
                #if the previous was an insertion, don't mark this one too
                if target[i-1] == "-":
                    continue
                else:
                    mod_inserts.append(i)
            except:
                mod_inserts.append(i)
        else:
            mod_score = mod_score + "0"
    return mod_score, mod_inserts

def analyze_false_positives(**kwargs):
    fig, ax = plt.subplots(layout='constrained')
    width = .3
    multiplier = 0
    for k, v in kwargs.items():
        fp_dict, p_count = Counter(v[0]), v[-1]
        y_values = []
        x_values = []
        x_order = ['p', 'ɓ', 'b', 'k', 'v','h', 'ɬ', 'ɮ']
        offset = width * multiplier
        for phoneme in x_order:
            if p_count[phoneme] == 0:
                continue
            else:
                norm_count = fp_dict[phoneme]/p_count[phoneme]
                #if norm_count > .50:
                y_values.append(norm_count)
                x_values.append(phoneme)
        print(x_values)
        x= np.arange(len(x_values))
        ax.bar(x + offset, y_values, width, label=k)
        multiplier += 1
    ax.legend()
    ax.set_xticks(x+width, x_values, fontsize=14)
    ax.set_xlabel('Mispronounced phoneme')
    ax.set_title('Percentage of mispronunciations missed by phoneme')
    ax.set_ylabel('Percentage of errors missed')
    plt.margins(x=0)
    plt.show()

def analyze_false_negatives(**kwargs):
    fig, ax = plt.subplots(layout='constrained')
    width = .3
    multiplier = 0
    for k, v in kwargs.items():
        fn_dict, p_counts = Counter(v[0]), v[1]
        y_values = []
        x_values = []
        #x_order = ['s', 'ʃ', 'ɦ', 'ǀ̬̃', 'ǃʰ', 'ǃ̃', 'ǃ̬', 'ǃ̬̃', 'ǁ̬']
        #uncomment for 25% 
        x_order = ['p', 't', 'k', 'g', 's', 'ʃ', 'ɦ', 'ǀ', 'ǀʰ', 'ǀ̬̃', 'ǃ', 'ǃʰ', 'ǃ̃', 'ǃ̃', 'ǃ̬', 'ǃ̬̃', 'ǁ', 'ǁʰ','ǁ̃', 'ǁ̬', '\\s']
        offset = width * multiplier
        for phoneme in x_order:
            if phoneme == 'EOS':
                continue
            norm_count = fn_dict[phoneme]/p_counts[phoneme]
            #if norm_count > .5:
            y_values.append(norm_count)
            x_values.append(phoneme)
        x= np.arange(len(x_values))
        print(x_values)
        ax.bar(x + offset, y_values, width, label=k)
        multiplier += 1
    ax.legend()
    ax.set_xticks(x+width, x_values, fontsize=14)
    ax.set_xlabel('Phoneme pronounced')
    ax.set_title('Percentage of correct phonemes labeled as mispronunciations')
    ax.set_ylabel('Percentage incorrectly rejected')
    plt.margins(x=0)
    plt.show()

   

#Function to make a bar graph of all phoneme errors identified
def graph_phoneme_errors(**kwargs):
    fig, ax = plt.subplots(layout='constrained')
    width = .25
    multiplier = 0
    for k, v in kwargs.items():
        found, missed = v[0], v[1]
        found = Counter(found)
        missed = Counter(missed)
        x = np.arange(len(found.keys()))
        labels = found.keys()
        coords = {}
        for phoneme, count in found.items():
            total_occurences = missed[phoneme]+count
            if phoneme in toIPA.keys():
                phoneme = toIPA[phoneme]
            coords[phoneme] = count/total_occurences
        coords = {key: coords[key] for key in x_order if key in coords}
        offset = width * multiplier
        ax.bar(x + offset, coords.values(), width, label=k)
        multiplier += 1
    ax.legend()
    ax.set_xticks(x+width, coords.keys())
    ax.set_xlabel("Phonemes")
    ax.set_title("Percentage of errors found by phoneme type")
    ax.set_ylabel("Percentage of errors found by model")
    plt.margins(x=.01)
    plt.show()
    plt.savefig("phoneme_errors.png")

#Function to graph the correct insertions identified by the model using a dictionary of the correct phonemes identified and the ones missed
def graph_c_inserts(**kwargs):
    linestyles = [":", "--", "-."]
    i= 0
    for k, v in kwargs.items():
        correct, counts = v[0], v[1]
        y_values = []
        x_values = []
        y1_values = []
        x1_values = []
        for x in list(correct.keys()):
            if x =='EOS':
                continue
            found = len(correct[x])
            missed = counts[x]
            total = found+missed
            if total == 0:
                continue
            else:
                x_values.append(x)
                y_values.append(found/total)
        #spl = make_interp_spline(x_values, y_values, k=3)
        #x_smooth = np.linspace(min(x_values), max(x_values), 200)
        #y_smooth = spl(x_smooth)
        #plt.plot(x_smooth, y_smooth, label=k, linestyle=linestyles[i]) #, alpha=.7)
        plt.plot(x_values, y_values, label=k, alpha= .5)#linestyle=linestyles[i])
        i += 1
    plt.xlabel("Index of error")
    plt.xticks(range(0,61,5))
    plt.legend()
    plt.ylabel("Percentage of errors found at that index")
    plt.title("Insert errors found by index")
    plt.show()
    plt.savefig("InsertIndexs.png")
    #for key in correct

def graph_inserts_by_phoneme(**kwargs):
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    width = .25
    #x = np.arange(len(x_order))
    for k,v in kwargs.items():
        correct, counts, missed = v[0], v[1], v[2]
        y_values = []
        new_labels = []
        sanity = 0
        for phoneme in x_order:
            found_count = 0
            missed_count = 0
            for i in range(len(correct)-1):
                found_p = correct[i].count(phoneme)
                missed_p = missed[i].count(phoneme)
                found_count += found_p
                missed_count += missed_p
            total = found_count+missed_count
            if total < 25:
                continue
                #y_values.append(np.nan)
            else:
                y_values.append(found_count/total)
                new_labels.append(phoneme+": "+str(total))
        x = np.arange(len(new_labels))
        offset = width * multiplier
        ax.bar(x + offset, y_values, width, label=k)
        multiplier +=1
    ax.legend()
    ax.set_xticks(x+width, new_labels, rotation=45)
    ax.set_xlabel("Phoneme following insertion")
    ax.set_title("Model Performance on Insertions by Phoneme Environment")
    ax.set_ylabel("Percentage of insertions identified")
    plt.margins(x=.01)
    plt.show()
    plt.savefig("phoneme_inserts.png")


def calculate_pr(df):
    df = preprocess(df)
    phoneme_count = 0
    true_negative = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct_insert = 0
    incorrect_insert = 0
    total_inserts =0
    found_perror = []
    missed_perror = []
    fn_phonemes = []
    incorrect_phonemes = []
    all_phonemes = []
    correct_phonemes = []
    count_miss = 0
    missed_inserts = {inserts_list: [] for inserts_list in range(60)}
    c_insert_errors = {inserts_list: [] for inserts_list in range(60)}
    c_insert_errors['EOS'] = []
    insert_counts = {key: 0 for key in range(60)}
    insert_counts['EOS'] = 0
    in_insert_errors = {inserts_list: [] for inserts_list in range(60)}
    in_insert_errors['EOS'] = []
    for i in range(len(df)):
        index, model_out, target, filename, db_score, insert = df['Unnamed: 0'][i], df['model_output'][i], df['target'][i], df['filename'][i], df['db_score'][i], df['insert'][i]
        db_score = choose_score(db_score)
        if db_score == False:
            continue
        for seq in problem_clicks:
            if seq in target:
                db_score = fix_clicks(target, db_score, seq)
        if "W" in target:
            db_score, target = fix_lw(target, db_score)
        aligned_tgt, aligned_mod = needleman_wunsch(target, model_out, match_score=1.0, mismatch_score=-1.0, indel_score=-1.0)
        mod_score, mod_inserts = get_score(aligned_mod, aligned_tgt)
        if len(db_score) != len(mod_score):
            print("ERR:", target, len(db_score), len(mod_score))
        #if "s" in target:
        #    print(aligned_tgt, '\n', aligned_mod, '\n', target, '\n')
        for i in range(len(db_score)):
            phoneme_count += 1
            if target[i] in toIPA.keys():
                all_phonemes.append(toIPA[target[i]])
            else:
                all_phonemes.append(target[i])
            #model correctly detects an error
            if db_score[i] == mod_score[i] == "0":
                true_negative += 1
                if target[i] in toIPA.keys():
                    incorrect_phonemes.append(toIPA[target[i]])
                    found_perror.append(target[i])
                else:
                    incorrect_phonemes.append(target[i])
                    found_perror.append(target[i])
            #model correctly marks good pronunciation
            elif db_score[i] == mod_score[i] == "1":
                true_positive += 1
                if target[i] in toIPA.keys():
                    correct_phonemes.append(toIPA[target[i]])
                else:
                    correct_phonemes.append(target[i])
            #model misses an error
            elif db_score[i] == "0" != mod_score[i]:
                false_positive += 1
                if target[i] in toIPA.keys():
                    missed_perror.append(toIPA[target[i]])
                    incorrect_phonemes.append(toIPA[target[i]])
                else:
                    missed_perror.append(target[i])
                    incorrect_phonemes.append(target[i])
            #model incorrectly marks an error
            else:
                false_negative += 1
                if target[i] in toIPA.keys():
                    fn_phonemes.append(toIPA[target[i]])
                    correct_phonemes.append(toIPA[target[i]])
                else:
                    fn_phonemes.append(target[i])
                    correct_phonemes.append(target[i])
       #print(target,'\n', model_out, '\n', insert, '\n', mod_inserts, '\n\n', sep="")
        for gins in insert:
            total_inserts+=1
            #dict showing count of insertions per index
            insert_counts[gins] += 1
            #if the insertion index is the last phoneme of the utterance also add to another list
            if gins == len(insert)-1:
                insert_counts['EOS'] += 1
            #if the model did not identify the insertion index
            if gins in mod_inserts:
                correct_insert += 1
                try:
                    if target[gins] in toIPA.keys():
                        c_insert_errors[gins].append(toIPA[target[gins]])
                    c_insert_errors[gins].append(target[gins])
                except:
                    c_insert_errors[gins].append('EOS')
            else:
                count_miss += 1
                try:
                    if target[gins] in toIPA.keys():
                        missed_inserts[gins].append(toIPA[target[gins]])
                    missed_inserts[gins].append(target[gins])
                except:
                    missed_inserts[gins].append('EOS')
        for ins in mod_inserts:
            if ins not in insert:
                #model predicted insert that isn't there
                incorrect_insert += 1
                try:
                    if target[gins] in toIPA.keys():
                        in_insert_errors[gins].append(toIPA[target[gins]])
                    in_insert_errors[ins].append(target[ins])
                except:
                    in_insert_errors[ins].append('EOS')
    phoneme_counts = Counter(all_phonemes)
    correct_phonemes = Counter(correct_phonemes)
    incorrect_phonemes = Counter(incorrect_phonemes)
    #print("FAR", false_positive/phoneme_count, "FRR", false_negative/phoneme_count)
    #print("Correct Inserts:", correct_insert, "Incorrect Inserts:", incorrect_insert, count_miss, total_inserts)
    #print("TN:", true_negative, "TP:", true_positive, "FP:", false_positive, "FN:", false_negative)
    #print("Precision:", true_positive/(true_positive + false_positive))
    #print("Recall:", true_positive/(true_positive + false_negative))
    #print("TNR:", true_negative/(true_negative + false_positive))
    return c_insert_errors, insert_counts, missed_inserts, found_perror, missed_perror, fn_phonemes, correct_phonemes, phoneme_counts, incorrect_phonemes

'''This function groups based on the utterance level
#Function to group each recording into a list
def create_groups(df):
    #list of sentences where teachers agree that the model is correct
    true_positive = pd.DataFrame()
    #list where model say there is an error, but teachers say there is not
    false_negative = pd.DataFrame()
    #both the model and the teacher detect an error
    true_negative = pd.DataFrame()
    #The model says it is correct, but teachers disagree
    false_positive = pd.DataFrame()
     #list where model thinks sentence is correct, but teachers say there is an insertion
    fp_insert = pd.DataFrame()
    #list where model thinks sentence is correct, but teachers say there is a phoneme error
    fp_phoneme = pd.DataFrame()
    #list where model thinks sentence is correct, but teachers marked both an insert and a phoneme error
    fp_both = pd.DataFrame()
    #loop through each transcription
    for i in range(0, len(df)):
        #provide variable name for each column for easier reference
        index, model_out, target, filename, db_score, insert = df['Unnamed: 0'][i], df['model_output'][i], df['target'][i], df['filename'][i], df['db_score'][i], df['insert'][i]
        #if model marks sentence as correct
        if target == model_out:
            #Teachers mark perfect phonemes
            if perfect_feedback(db_score)==True:
                #Teachers note no inserts
                if insert == []:
                    true_positive = true_positive._append(df.iloc[i], ignore_index=True)
                else:
                    #model misses a sound insertion
                    fp_insert = fp_insert._append(df.iloc[i], ignore_index=True)
                    false_positive = false_positive._append(df.iloc[i], ignore_index=True)
            #model misses a phoneme error
            else:
                fp_phoneme = fp_phoneme._append(df.iloc[i], ignore_index=True)
                false_positive = false_positive._append(df.iloc[i], ignore_index=True)
                if insert != []:
                    fp_both = fp_both._append(df.iloc[i], ignore_index=True)
        #model marks sentence as incorrect
        else:
            #teachers mark sentence as correct
            if perfect_feedback(db_score)==True and insert == []:
                false_negative = false_negative._append(df.iloc[i], ignore_index=True)
            else:
                true_negative = true_negative._append(df.iloc[i], ignore_index=True)
                if len(target) != len(db_score[0]):
                    print("Error in alignment", filename, len(db_score[0]), target)
    return true_positive, true_negative, false_positive, false_negative, fp_insert, fp_phoneme, fp_both
'''
mdd_df = pd.read_csv("scores/mdd_transcripts_w_scores.csv")
nchlt_df = pd.read_csv("scores/nchlt_transcripts_w_scores.csv")
com_df = pd.read_csv("scores/com_transcripts_w_scores.csv")
print("MDD model:")
mdd = calculate_pr(mdd_df)
print("NCHLT Model:")
nchlt = calculate_pr(nchlt_df)
print("Com model:")
com = calculate_pr(com_df)

#graph_c_inserts(L2=mdd, NCHLT=nchlt, L2NCHLT=com)

#graph_inserts_by_phoneme(L2=mdd, NCHLT=nchlt, L2NCHLT=com)

analyze_false_positives(L2=mdd[4:], NCHLT=nchlt[4:], L2NCHLT=com[4:])
#analyze_false_negatives(L2=mdd[5:7], NCHLT=nchlt[5:7], L2NCHLT=com[5:7])

#graph_phoneme_errors(L2=mdd[3:5], NCHLT=nchlt[3:5], L2NCHLT=com[3:5])

#dataframe_names = ['true_positive', 'true_negative', 'false_positive', 'false_negative', 'fp_insert', 'fp_phoneme', 'fp_both']

#groups = create_groups(model_df)

#for i in range(len(groups)):
#    filename = sys.argv[2]+"/"+dataframe_names[i]+".csv"
#    groups[i].to_csv(filename, index=False)


'''
#Print out statements for precision and recall

tp = len(true_positive)
fp = len(false_positive)
fn = len(false_negative)
tn = len(true_negative)
print("Both teachers and model agree the pronunciation is correct (True Positive):", len(true_positive))
print("Model misses a phoneme insertion", len(fp_insert))
print("Model misses a mispronounced phoneme:", len(fp_phoneme))
print("Model misses a mispronounced phoneme and an insertion:", len(fp_both))
print("Missed phoneme and insertion errors from the model (False Positive):", len(false_positive))
print("Model says there is an error, but teachers disagree (False Negative):", len(false_negative))
print("Both agree that there is an error (True Negative):", len(true_negative))
#print(model_df)

print("Precision:", tp/(tp+fp))
print("Recall:", tp/(tp+fn))
'''
