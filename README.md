We started with the skeleton from https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/nlp
Credits to Guillaume Genthial and Olivier Moindrot

## Setup
```
./setup.sh
```

## Generate our final model:
First generate wrong endings as follows:
```
python wrong_endings_generation.py --generation_method <method>
```
wrong_ending_generation_antonyms.py
wrong_ending_generation_antonyms_nltk.py
wrong_ending_generation_shuffle.py
```

Then:
```
python build_dataset.py
python build_vocab.py
python train.py
```
