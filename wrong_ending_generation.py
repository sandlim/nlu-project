import pandas as pd
import re
import random
from collections import Counter
import os

def load_data(path, is_validation=False):
    df = pd.read_csv(path)
    return df

def save_dataset(dataset, save_path):
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_path))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Export the dataset
    dataset.to_csv(save_path, index=False)
    print("- done.")


def generate_endings(path, store_path):
    word_counts = Counter()
    df = load_data(path)
    antonyms = dict([
        ("day", "night"),
        ("was", "wasn't"),
        ("will", "won't"),
        ("did", "didn't"),
        ("could", "couldn't"),
        ("home", "abroad"),
        ("would", "wouldn't"),
        ("friends", "enemies"),
        ("friend", "enemy"),
        ("with", "without"),
        ("bought", "sold"),
        ("loved", "hated"),
        ("started", "ended"),
        ("first",  "last"),
        ("good", "bad"),
        ("always", "never"),
        ("happy", "sad"),
        ("great", "awful"),
        ("party", "funeral"),
        ("something", "nothing"),
        ("most", "least"),
        ("nervous", "calm")
    ])
    invertible_words = set(antonyms.keys()) | set(antonyms.values())
    occurences = dict([(word, []) for word in invertible_words])

    s5 = df["sentence5"]
    for i, sentence in enumerate(s5):
        # if i < 50:
        #     print(str(i) + " " + sentence)

        for word in re.findall(r"[\w']+", sentence):
            if word in invertible_words:
                occurences[word] += [i]
            # if i < 50:
            #     print(word)
            if word in word_counts:  # collect word frequencies (used later to prune vocabulary)
                word_counts[word] += 1
            else:
                word_counts[word] = 1

    stories_w = pd.DataFrame(columns=['sentence1', 'sentence2',
                                     'sentence3', 'sentence4', 'sentence5',
                                     'label'])

    index = 0
    for k, v in antonyms.items():
        w1, w2 = (k, v) if word_counts[k] > word_counts[v] else (v, k)
        w1_count, w2_count = (word_counts[w1], word_counts[w2])
        w1_num_to_invert = int(w2_count * (w2_count / w1_count))
        w2_num_to_invert = w2_count
        # print(occurences[w1])
        random.shuffle(occurences[w1])
        w1_invert_indices = occurences[w1][:w1_num_to_invert]
        w2_invert_indices = occurences[w2]
        for i in w1_invert_indices:
            story = df.iloc[i]
            stories_w.loc[index] = [
                                    story['sentence1'],
                                    story['sentence2'],
                                    story['sentence3'],
                                    story['sentence4'],
                                    str(story['sentence5']).replace(w1, w2, 1),
                                    0
                                   ]
            index += 1
        for i in w2_invert_indices:
            story = df.iloc[i]
            stories_w.loc[index] = [
                                    story['sentence1'],
                                    story['sentence2'],
                                    story['sentence3'],
                                    story['sentence4'],
                                    str(story['sentence5']).replace(w2, w1, 1),
                                    0
                                   ]
            index += 1

    save_dataset(stories_w,store_path)


def main():
    train_path = './data/train_stories.csv'
    generate_endings(train_path, './data/wrong_endings.csv')


if __name__ == "__main__":
    main()

