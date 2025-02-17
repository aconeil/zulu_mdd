import string 
import sys

def zulu2ipa(text):
    transcribe = {"ngc":"1","ngq":"2", "ngx":"3", "bh" : "?", "mb" : "M", "b" :"ɓ","?":"b",
              "kh" :"K", "ph" :"P","th" :"T", "hh" :"ɦ", "hl" : "ɬ", "dl" : "ɮ", "ng" : "ŋ",
              "ny" : "ɲ", "sh" : "ʃ", "ch" : "C", "gc" : "4", "nc" : "7", "qh" : "Q",
              "xh" : "X", "gq" : "6", "nq" : "9", "gx" : "5", "nx" : "8", "j" : "ʤ", 
              "y" : "j", "r" : "ʁ", "c" : "c", "q" : "q", "x" : "x", "k" : "k", "t":"t", 
              "p":"p", "a":"a", "e":"e", "i":"i", "o":"o", "u":"u", 
              "d":"d", "f":"f", "g":"g", "h":"h", "l":"l", "m":"m", "n":"n", "s":"s", "v":"v", 
                  "w":"w", "z":"z", "M":"mb", "lw":"W"}
    text = text.lower()
    remove_punct = str.maketrans('', '', string.punctuation)
    text = text.translate(remove_punct)
    words = text.split(" ")
    #replace zulu characters with ipa characters based on the above list
    for zulu, ipa in transcribe.items():
        words = [word.replace(zulu, ipa) for word in words]
    return "|".join(words)
    #return ipa_text
'''Uncomment for command line use
file = open(sys.argv[1])

for line in file.readlines():
    ipa = zulu2ipa(line, transcribe)
    #print(ipa)
'''
