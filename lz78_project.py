from lz78 import Sequence, LZ78Encoder, CharacterMap, BlockLZ78Encoder, LZ78SPA
from lz78 import encoded_sequence_from_bytes, spa_from_bytes
import numpy as np
import lorem
import requests
from sys import stdout
from os import makedirs

import pandas as pd



def train_spa(seq, spa):
    encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
    seq_len = len(seq)
    train_logloss = spa.train_on_block(encoded_seq) / (seq_len)
    return train_logloss

def test_seq (seq, spa):
    encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
    seq_len = len(seq)
    test_logloss = spa.compute_test_loss(encoded_seq, include_prev_context=False) / seq_len #yes or no?
    return test_logloss


def main():

    # read train.csv
    train_path = "/Users/yasmineomri/Downloads/GUE/mouse/0/train.csv"
    train_data = pd.read_csv(train_path)

    # create list of spas based on number of labels: (spa_0 and spa_1 for labels 0, 1)
    ALPHABET_SIZE = 4
    spa_0 = LZ78SPA(ALPHABET_SIZE)
    spa_1 = LZ78SPA(ALPHABET_SIZE)

    logloss_0 = []
    logloss_1 = []

    for row in train_data.itertuples():
        seq = row[1]
        label = row[2]
        if label == 1:
            logloss_1.append(train_spa(seq, spa_1))

        elif label == 0:
            logloss_0.append(train_spa(seq, spa_0))

    # print(logloss_0)
    # print(logloss_1)

    # read test.csv  
    # test_path = "/Users/yasmineomri/Downloads/GUE/mouse/0/test.csv"

    test_path = "/Users/yasmineomri/Downloads/GUE/mouse/1/test.csv"
    test_data = pd.read_csv(test_path)

    # for every test seq,
    # run it through all spas
    # classification = label associated with lowest loss spa
    # check classification against ground truth
    # compute accuracy (of all test runs)
    nb_correct = 0
    nb_test_total = 0
    for row in train_data.itertuples():
        seq = row[1]
        correct_label = row[2]
        nb_test_total += 1

        spa0_logloss = test_seq (seq, spa_0)
        spa1_logloss = test_seq (seq, spa_1)
        
        
        predicted_label = 0 if spa0_logloss < spa1_logloss else 1
        
        if predicted_label == correct_label:
            nb_correct += 1 

    accuracy = nb_correct / nb_test_total 
    print("The total number of tested sequences is: ", nb_test_total)
    print("The accuracy of this classifier is: ", accuracy)
        




if __name__ == "__main__":
    main()


