import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk.data
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

#load the data
train_dat = pd.read_csv('./data/train_stories.csv')

#generate wrong sentences
train_dat['wsentence5'] = train_dat['sentence5']
for i in range(len(train_dat['wsentence5'])):
    train_dat['wsentence5'][i] = negateSentence(train_dat['sentence5'][i])

#check the generated wrong sentences are all negated. Sometimes NLTK cannot identify verb correctly=> add not.
train_dat['w2sentence5'] = train_dat['wsentence5']

for i in range(len(train_dat['w2sentence5'])):
    train_dat['w2sentence5'][i]= train_dat['w2sentence5'][i].lstrip()
    if train_dat['w2sentence5'][i].replace(' .','.').replace(' ,',',') == train_dat['sentence5'][i]:
        train_dat['w2sentence5'][i] = train_dat['sentence5'][i].replace('.','')  + " not."
    else:
        train_dat['wsentence5'][i] == train_dat['wsentence5'][i]

#create two choices: randomize s5 and w2s5
choice = pd.DataFrame(columns=['RandomFifthSentenceQuiz1','RandomFifthSentenceQuiz2'])

choice["RandomFifthSentenceQuiz1"] = train_dat["sentence5"]
choice["RandomFifthSentenceQuiz2"] = train_dat["w2sentence5"]

np.random.seed(37)

choice = choice.values
#choice.shape (88161, 2)

_ = [np.random.shuffle(i) for i in choice]

choice = pd.DataFrame(choice,
                      columns=['RandomFifthSentenceQuiz1','RandomFifthSentenceQuiz2'])

train_dat["RandomFifthSentenceQuiz1"] = choice["RandomFifthSentenceQuiz1"]
train_dat["RandomFifthSentenceQuiz2"] = choice["RandomFifthSentenceQuiz2"]

train_dat["AnswerRightEnding"] = np.where(train_dat['RandomFifthSentenceQuiz1']==train_dat['sentence5'], '1', '2')


del train_dat['sentence5']
del train_dat['wsentence5']
del train_dat['w2sentence5']

train_dat.to_csv("./data/wrong_ending_generation_nltk.csv", sep=',', encoding='utf-8',index = False)