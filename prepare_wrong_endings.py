import os
from build_dataset import load_dataset, save_dataset, tokenize

if __name__ == "__main__":

    # Check that the dataset exists
    path_dataset_generated = 'data/wrong_endings.csv'
    msg = "{} file not found. Make sure you have downloaded the right dataset".format(
        path_dataset_generated)
    assert os.path.isfile(path_dataset_generated), msg

    # Load the dataset into memory
    print("Loading dataset into memory (and rearranging it)...")
    dataset_generated = load_dataset(path_dataset_generated, generated=True)
    print("- loaded generated stories.")
    print("- done.")

    # Tokenization
    print("Tokenization...")
    dataset_generated = tokenize(dataset_generated)
    print("- tokenized generated set.")
    print("- done.")


    # Save the datasets to files
    save_dataset(dataset_generated, os.path.join('data', 'dev_split', 'train', 'stories_generated.csv'))
