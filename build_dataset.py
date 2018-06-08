"""Read, split and save the dataset for our model"""

import os
import pandas as pd
import nltk
import numpy as np


def load_dataset(path_csv, is_validation=False, generated=False, is_test=False):
    """Loads dataset into memory from csv file"""

    if is_validation:
        val_df = pd.read_csv(
            path_csv,
            usecols=[
                'InputSentence1', 'InputSentence2', 'InputSentence3',
                'InputSentence4', 'RandomFifthSentenceQuiz1',
                'RandomFifthSentenceQuiz2', 'AnswerRightEnding'
            ])
        df1 = pd.DataFrame(columns=[
            'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5',
            'label'
        ])
        df2 = pd.DataFrame(columns=[
            'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5',
            'label'
        ])
        for index, row in val_df.iterrows():
            first_ending = row['RandomFifthSentenceQuiz1']
            second_ending = row['RandomFifthSentenceQuiz2']
            df1.loc[index] = [
                row['InputSentence1'], row['InputSentence2'],
                row['InputSentence3'], row['InputSentence4'], first_ending,
                2 - row['AnswerRightEnding']
            ]
            df2.loc[index] = [
                row['InputSentence1'], row['InputSentence2'],
                row['InputSentence3'], row['InputSentence4'], second_ending,
                row['AnswerRightEnding'] - 1
            ]
        return df1, df2
    elif is_test:
        val_df = pd.read_csv(
            path_csv,
            usecols=[
                'InputSentence1', 'InputSentence2', 'InputSentence3',
                'InputSentence4', 'RandomFifthSentenceQuiz1',
                'RandomFifthSentenceQuiz2'
            ])
        df1 = pd.DataFrame(columns=[
            'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5',
            'label'
        ])
        df2 = pd.DataFrame(columns=[
            'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5',
            'label'
        ])
        for index, row in val_df.iterrows():
            first_ending = row['RandomFifthSentenceQuiz1']
            second_ending = row['RandomFifthSentenceQuiz2']
            df1.loc[index] = [
                row['InputSentence1'], row['InputSentence2'],
                row['InputSentence3'], row['InputSentence4'], first_ending,
                0
            ]
            df2.loc[index] = [
                row['InputSentence1'], row['InputSentence2'],
                row['InputSentence3'], row['InputSentence4'], second_ending,
                0
            ]
        return df1, df2
    else:
        if generated:
            df = pd.read_csv(
                path_csv,
                usecols=[
                    'sentence1', 'sentence2', 'sentence3', 'sentence4',
                    'sentence5', 'label'
                ])
        else:
            df = pd.read_csv(
                path_csv,
                usecols=[
                    'sentence1', 'sentence2', 'sentence3', 'sentence4',
                    'sentence5'
                ])
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
    return " ".join([word.lower() for word in nltk.word_tokenize(sentence)])


def tokenize(ds):
    return ds.applymap(tokenize_sentence)


if __name__ == "__main__":
    nltk.download('punkt')

    # Check that the dataset exists
    path_dataset = 'data/train_stories.csv'
    path_dataset_val = 'data/cloze_test_val.csv'
    path_dataset_test = 'data/cloze_test_spring2016-test.csv'
    path_dataset_test_nlu18 = 'data/test_nlu18.csv'
    path_dataset_generated = 'data/wrong_endings.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(
        path_dataset)
    assert os.path.isfile(path_dataset), msg
    assert os.path.isfile(path_dataset_val), msg
    assert os.path.isfile(path_dataset_test), msg
    assert os.path.isfile(path_dataset_generated), msg

    # Load the dataset into memory
    print("Loading dataset into memory (and rearranging it)...")
    train_dataset = load_dataset(path_dataset)
    print("- loaded training set.")
    dataset_val1, dataset_val2 = load_dataset(
        path_dataset_val, is_validation=True)
    print("- loaded (and rearranged) validation set.")
    dataset_test1, dataset_test2 = load_dataset(
        path_dataset_test, is_validation=True)
    print("- loaded (and rearranged) test set.")
    dataset_test_nlu18_1, dataset_test_nlu18_2 = load_dataset(
        path_dataset_test, is_test=True)
    print("- loaded (and rearranged) test_nlu18 set.")
    dataset_generated = load_dataset(path_dataset_generated, generated=True)
    print("- loaded generated stories.")
    print("- done.")

    # Tokenization
    print("Tokenization...")
    dataset_val1 = tokenize(dataset_val1)
    dataset_val2 = tokenize(dataset_val2)
    print("- tokenized validation set.")
    dataset_test1 = tokenize(dataset_test1)
    dataset_test2 = tokenize(dataset_test2)
    print("- tokenized test set.")
    dataset_test_nlu18_1 = tokenize(dataset_test_nlu18_1)
    dataset_test_nlu18_2 = tokenize(dataset_test_nlu18_2)
    print("- tokenized test_nlu18 set.")
    train_dataset = tokenize(train_dataset)
    print("- tokenized training set.")
    dataset_generated = tokenize(dataset_generated)
    print("- tokenized generated set.")

    print("- done.")

    inds = np.random.permutation(dataset_val1.index.values.tolist())
    split = int(0.8 * len(dataset_val1))
    dev_inds = list(inds[:split])
    val_inds = list(inds[split:])
    dev_split_dataset1 = dataset_val1.loc[dev_inds]
    val_split_dataset1 = dataset_val1.loc[val_inds]
    dev_split_dataset2 = dataset_val2.loc[dev_inds]
    val_split_dataset2 = dataset_val2.loc[val_inds]

    # Save the datasets to files
    save_dataset(train_dataset,
                 os.path.join('data', 'dev_split', 'train', 'stories.csv'))
    save_dataset(
        dataset_generated,
        os.path.join('data', 'dev_split', 'train', 'stories_generated.csv'))
    save_dataset(dev_split_dataset1,
                 os.path.join('data', 'dev_split', 'dev', 'stories1.csv'))
    save_dataset(dev_split_dataset2,
                 os.path.join('data', 'dev_split', 'dev', 'stories2.csv'))
    save_dataset(val_split_dataset1,
                 os.path.join('data', 'dev_split', 'val', 'stories1.csv'))
    save_dataset(val_split_dataset2,
                 os.path.join('data', 'dev_split', 'val', 'stories2.csv'))
    save_dataset(dataset_test1,
                 os.path.join('data', 'dev_split', 'test', 'stories1.csv'))
    save_dataset(dataset_test2,
                 os.path.join('data', 'dev_split', 'test', 'stories2.csv'))
    save_dataset(dataset_test_nlu18_1,
                 os.path.join('data', 'dev_split', 'test_nlu18', 'stories1.csv'))
    save_dataset(dataset_test_nlu18_2,
                 os.path.join('data', 'dev_split', 'test_nlu18', 'stories2.csv'))
