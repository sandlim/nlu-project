We started with the skeleton from https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/nlp
Credits to Guillaume Genthial and Olivier Moindrot

## Setup
```
./setup.sh
```

## Usage
First generate wrong endings as follows:
```
python wrong_endings_generation.py --generation_method <method>
```

Then:
```
python build_dataset.py
python build_vocab.py
python train.py
```
