import pandas as pd
import random
from build_dataset import save_dataset, tokenize


def load_data(path, is_validation=False):
    df = pd.read_csv(path, usecols=[
                         'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5'
                     ])
    return df


def generate_endings_shuffle(df: pd.DataFrame):
    df_new = df.copy()
    indices = list(range(len(df)))
    while any([e == i for e, i in enumerate(indices)]):
        random.shuffle(indices)
    for i, ind in enumerate(indices):
        df_new.iloc[i]['sentence5'] = df.iloc[ind]['sentence5']
    df_new['label'] = 0
    return df_new


def main():
    random.seed(42)
    train_path = './data/train_stories.csv'
    df_train = load_data(train_path)
    df = generate_endings_shuffle(df_train)
    df = tokenize(df)
    save_dataset(df, './data/dev_split/train/shuffled_endings.csv')


if __name__ == "__main__":
    main()

