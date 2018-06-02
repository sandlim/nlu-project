"""Read, split and save the dataset for our model"""

import os
import pandas as pd
import nltk
import numpy as np


def load_dataset(path_csv, is_validation=False):
    """Loads dataset into memory from csv file"""

    if is_validation:
        val_df = pd.read_csv(path_csv, usecols=['InputSentence1',
                                                'InputSentence2',
                                                'InputSentence3',
                                                'InputSentence4',
                                                'RandomFifthSentenceQuiz1',
                                                'RandomFifthSentenceQuiz2',
                                                'AnswerRightEnding'])
        df_correct = pd.DataFrame(columns=['sentence1', 'sentence2',
                                   'sentence3', 'sentence4', 'sentence5',
                                   'label'])
        df_wrong = pd.DataFrame(columns=['sentence1', 'sentence2',
                                   'sentence3', 'sentence4', 'sentence5',
                                   'label'])
        for index, row in val_df.iterrows():
            correct_ending = (row['RandomFifthSentenceQuiz1'] if row['AnswerRightEnding'] == 1
                              else row['RandomFifthSentenceQuiz2'])
            wrong_ending = (row['RandomFifthSentenceQuiz1'] if row['AnswerRightEnding'] == 2
                            else row['RandomFifthSentenceQuiz2'])
            df_correct.loc[index] = [row['InputSentence1'],
                                 row['InputSentence2'],
                                 row['InputSentence3'],
                                 row['InputSentence4'],
                                 correct_ending,
                                 1]
            df_wrong.loc[index] = [row['InputSentence1'],
                                     row['InputSentence2'],
                                     row['InputSentence3'],
                                     row['InputSentence4'],
                                     wrong_ending,
                                     0]
        return df_correct, df_wrong

    else:
        df = pd.read_csv(path_csv, usecols=['sentence1', 'sentence2',
                                            'sentence3', 'sentence4', 'sentence5'])
        # train endings are all correct
        df['label'] = 1

        return df


def save_dataset(dataset, save_path):
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_path))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Export the dataset
    dataset.to_csv(save_path, index=False)
    print("- done.")


def tokenize_sentence(sentence):
    if not isinstance(sentence, str):
        return sentence
    return " ".join([word for word in nltk.word_tokenize(sentence)])


def tokenize(ds):
    return ds.applymap(tokenize_sentence)


if __name__ == "__main__":
    # Check that the dataset exists
    path_dataset = 'data/train_stories.csv'
    path_dataset_val = 'data/cloze_test_val.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg
    assert os.path.isfile(path_dataset_val), msg

    # Load the dataset into memory
    print("Loading dataset into memory (and rearranging it)...")
    dataset = load_dataset(path_dataset)
    print("- loaded training set.")
    dataset_val_c, dataset_val_w = load_dataset(path_dataset_val, is_validation=True)
    print("- loaded (and rearranged) validation set.")
    print("- done.")

    # Tokenization
    print("Tokenization...")
    dataset_val_c = tokenize(dataset_val_c)
    dataset_val_w = tokenize(dataset_val_w)
    print("- tokenized validation set.")
    dataset = tokenize(dataset)
    print("- tokenized training set.")
    print("- done.")

    train_dataset = dataset
    inds = np.random.permutation(dataset_val_c.index.values.tolist())
    split = int(0.8 * len(dataset_val_c))
    dev_inds = list(inds[:split])
    val_inds = list(inds[split:])
    dev_split_dataset_c = dataset_val_c.loc[dev_inds]
    val_split_dataset_c = dataset_val_c.loc[val_inds]
    dev_split_dataset_w = dataset_val_w.loc[dev_inds]
    val_split_dataset_w = dataset_val_w.loc[val_inds]

    # Save the datasets to files
    save_dataset(train_dataset, os.path.join('data', 'dev_split', 'train', 'stories.csv'))
    save_dataset(dev_split_dataset_c, os.path.join('data', 'dev_split', 'dev', 'stories_c.csv'))
    save_dataset(dev_split_dataset_w, os.path.join('data', 'dev_split', 'dev', 'stories_w.csv'))
    save_dataset(val_split_dataset_c, os.path.join('data', 'dev_split', 'val', 'stories_c.csv'))
    save_dataset(val_split_dataset_w, os.path.join('data', 'dev_split', 'val', 'stories_w.csv'))
