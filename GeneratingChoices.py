import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
from itertools import chain
from nltk.stem.wordnet import WordNetLemmatizer

def get_antonyms(input_lemma):
    antonyms = []
    for syn in wn.synsets(input_lemma):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms


def changeAdj(i, words, tagged, isNegated):
    if tagged[i][1] == 'JJ' or tagged[i][1] == 'RB':
        antonyms = get_antonyms(words[i])
        if len(antonyms) > 0:
            new_word = antonyms[0]
            isNegated = True
        else:
            new_word = words[i]

    else:
        new_word = words[i]

    return new_word, isNegated


def changeVerb(i, words, tagged, isNegated):
    if words[i] == 'was' or words[i] == 'were':
        new_verb = words[i] + " " + "not"
        isNegated = True


    elif tagged[i][1] == 'VBD' and words[i] != 'was' and words[i] != 'were':
        new_verb = "did not" + " " + WordNetLemmatizer().lemmatize(words[i], 'v')
        isNegated = True
    else:
        new_verb = words[i]

    return new_verb, isNegated


def negateSentence(sentence):
    words = word_tokenize(sentence)
    tagged = nltk.pos_tag(words)
    isNegated = False
    wordTypes = [word[1] for word in tagged]
    hasAdj = 'JJ' in wordTypes or 'RB' in wordTypes

    output = ""
    for i in range(0, len(words)):
        if hasAdj and not isNegated:
            new_word, isNegated = changeAdj(i, words, tagged, isNegated)
        elif not hasAdj and not isNegated:
            new_word, isNegated = changeVerb(i, words, tagged, isNegated)
        #   elif i==len(words)  and not isNegated:
        #       new_word = words[i] + "not"
        else:
            new_word = words[i]
        output = output + " " + new_word
    return output

train_dat = pd.read_csv('./data/train_stories.csv')

train_dat['wsentence5'] = train_dat['sentence5']
for i in range(len(train_dat['wsentence5'])):
    train_dat['wsentence5'][i] = negateSentence(train_dat['sentence5'][i])


choice = pd.DataFrame(columns=['Option1','Option2'])

choice["Option1"] = train_dat["sentence5"]
choice["Option2"] = train_dat["wsentence5"]

choice = choice.values
#choice.shape (88161, 2)
_ = [np.random.shuffle(i) for i in choice]

choice = pd.DataFrame(choice,
                      columns=['Option1','Option2'])

train_dat["Option1"] = choice["Option1"]
train_dat["Option2"] = choice["Option2"]

train_dat["Correct"] = np.where(train_dat['Option1']==train_dat['sentence5'], '1', '2')

train_dat.to_csv("./data/train_dat_wrongchoice.csv", sep=',', encoding='utf-8',index = False)