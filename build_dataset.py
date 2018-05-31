"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys
import pandas as pd


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
        df = pd.DataFrame(columns=['sentence1','sentence2',
                                   'sentence3','sentence4','sentence5',
                                   'label'])
        for index, row in val_df.iterrows():
            correct_ending = row['RandomFifthSentenceQuiz1'] if row['AnswerRightEnding'] == 1 else row['RandomFifthSentenceQuiz2']
            wrong_ending = row['RandomFifthSentenceQuiz1'] if row['AnswerRightEnding'] == 2 else row['RandomFifthSentenceQuiz2']
            df.loc[2*index] = [row['InputSentence1'],
                               row['InputSentence2'],
                               row['InputSentence3'], 
                               row['InputSentence4'],
                               correct_ending,
                               1]
            df.loc[2*index + 1] = [row['InputSentence1'],
                                   row['InputSentence2'],
                                   row['InputSentence3'], 
                                   row['InputSentence4'],
                                   wrong_ending,
                                   0]
                               



    else:
        df = pd.read_csv(path_csv, usecols=['sentence1','sentence2',
                                            'sentence3','sentence4','sentence5']) 
        # train endings are all correct
        df['label'] = 1 


    return df


def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    dataset.to_csv(os.path.join(save_dir, 'stories.csv'))
    print("- done.")


if __name__ == "__main__":
    # Check that the dataset exists
    path_dataset = 'data/train_stories.csv'
    path_dataset_val = 'data/cloze_test_val.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg
    assert os.path.isfile(path_dataset_val), msg

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(path_dataset)
    dataset_val = load_dataset(path_dataset_val, is_validation=True)
    print("- done.")

    # Split the dataset into train, dev and split (dummy split with no shuffle)
    # train_dataset = dataset[:int(0.7*len(dataset))]
    train_dataset = dataset
    dev_dataset = dataset_val
    # test_dataset = dataset[int(0.85*len(dataset)):]

    # Save the datasets to files
    save_dataset(train_dataset, 'data/dev_split/train')
    save_dataset(dev_dataset, 'data/dev_split/val')
    # save_dataset(dev_dataset, 'data/kaggle/dev')
    # save_dataset(test_dataset, 'data/kaggle/test')

