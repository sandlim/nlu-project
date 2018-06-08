import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk.data
from nltk.stem.wordnet import WordNetLemmatizer

from build_dataset import tokenize


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


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


# load the data
train_dat = pd.read_csv('./data/train_stories.csv', usecols=[
    'sentence1', 'sentence2', 'sentence3', 'sentence4',
    'sentence5'
])

# generate wrong sentences
train_dat['wsentence5'] = train_dat['sentence5']
for i in range(len(train_dat['wsentence5'])):
    train_dat['wsentence5'][i] = negateSentence(train_dat['sentence5'][i])

# check the generated wrong sentences are all negated. Sometimes NLTK cannot identify verb correctly=> add not.
train_dat['w2sentence5'] = train_dat['wsentence5']

for i in range(len(train_dat['w2sentence5'])):
    train_dat['w2sentence5'][i] = train_dat['w2sentence5'][i].lstrip()
    if train_dat['w2sentence5'][i].replace(' .', '.').replace(' ,', ',') == train_dat['sentence5'][i]:
        train_dat['wsentence5'][i] = train_dat['sentence5'][i].replace('.', '') + " not."
    else:
        train_dat['sentence5'] = train_dat['wsentence5']

del train_dat['w2sentence5']
del train_dat['wsentence5']
train_dat['label'] = 0

train_dat = tokenize(train_dat)
train_dat.to_csv("./data/dev_split/train/antonym_endings_nltk.csv", sep=',', encoding='utf-8', index=False)
