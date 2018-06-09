We started with the skeleton from https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/nlp
Credits to Guillaume Genthial and Olivier Moindrot

## Setup
virtualenv needs to be available.
```
./setup.sh
```

## Generate our final model:
First generate wrong endings as follows:
```
python wrong_endings_generation_<generation-method>.py
```
wrong_ending_generation_antonyms.py
wrong_ending_generation_antonyms_nltk.py
wrong_ending_generation_shuffle.py
```

Then:
```
python build_dataset.py
python build_vocab.py
```


## Steps to reach out final model

Execute in order:
```
python generator_pretrain.py --overwrite
python generator_train.py --overwrite --restore_dir experiments/generator_pre-model/best_weights
python generator_infer.py
python train.py --overwrite
python infer.py  
```

