from lz78 import Sequence, LZ78Encoder, CharacterMap, BlockLZ78Encoder, LZ78SPA
from lz78 import encoded_sequence_from_bytes, spa_from_bytes
import numpy as np
import lorem
import requests
from sys import stdout
from os import makedirs
import time
import pandas as pd
import random
import argparse
import math
import itertools
from memory_profiler import profile
import os



# Training hyperparameters:
INCLUDE_PREV_CONTEXT = None
GAMMA = None
NB_TRAIN_ITERATIONS = None
HANDLE_N_SETTING = None 
RATIO_PRETRAIN_TRAIN = None # nb of pretrained sequences / nb train sequences


# include_prev_contexts = True
# gammas = 3
# nb_train_iterations = 10
# handle_N_settings = "remove"
# ratio_pretrain_train = 0.25

import argparse

def parse_set(input_str):
    """
    Converts a comma-separated string into a Python set.
    Example: "{0.1, 0.5, 1.0}" -> {0.1, 0.5, 1.0}
    """
    input_str = input_str.strip("{}")  # Remove the surrounding braces
    return set(float(x.strip()) if x.replace('.', '', 1).isdigit() else x.strip()
               for x in input_str.split(",") if x)


def parse_bool(input_str):
    # Strip the braces and split the string into components
    input_str = input_str.strip("{}")
    items = input_str.split(",")
    # Trim and convert to appropriate types
    return {item.strip() == "True" for item in items}

# Hyperparameters
# include_prev_contexts = {True, False}
# gammas = {0.1, 0.33, 0.5, 0.75, 1, 3, 5}
# # gammas = {0.1, 0.33, 0.5, 0.75, 1, 2, 3, 4, 5}

# nb_train_iterations = {1, 3, 5, 7, 10}
# handle_N_settings = {"remove"} #TODO smaller study on random
# ratio_pretrain_train = {0, 0.01, 0.1, 0.25}

# ratio_pretrain_train = {0, 0.005, 0.01, 0.05, 0.1, 0.5}
# handle_N_settings = {"remove", "random", "expand"}


# include_prev_contexts = {False}
# gammas = {4, 0.33}
# nb_train_iterations = {1, 6}
# handle_N_settings = {"remove"}
# ratio_pretrain_train = {0, 0.01}



# INCLUDE_PREV_CONTEXT = False
# GAMMA = 4
# NB_TRAIN_ITERATIONS = 6
# HANDLE_N_SETTING = "remove"
# RATIO_PRETRAIN_TRAIN = 0.1 # nb of pretrained sequences / nb train sequences



def pretrain_spa(seq, spa, nb_pretrain_symbols):
    global INCLUDE_PREV_CONTEXT
    if nb_pretrain_symbols == 0:
        return
    # Split the sequence into individual elements by newline
    elements = seq.splitlines()
    
    # Determine the number of elements to use based on the specified percentage
    # num_elements = int(len(elements) * (percentage / 100))
    len_pretrain_seq = len(elements[0])
    nb_pretrain_seqs = math.ceil(nb_pretrain_symbols / len_pretrain_seq)

    selected_elements = elements[:nb_pretrain_seqs]
    selected_elements[-1] = selected_elements[-1][0: nb_pretrain_symbols % len_pretrain_seq]

    for element in selected_elements:
        # Encode each individual element
        encoded_seq = Sequence(element, charmap=CharacterMap("ACGT"))
        seq_len = len(element)

        if (seq_len == 0):
            print("Warning: seq_len = 0 during pretraining. Exited pretrain.")
            return
        # Compute log-loss for each label's SPA
        for index in range(len(spa)):
            spa[index].train_on_block(encoded_seq, include_prev_context=INCLUDE_PREV_CONTEXT) / seq_len


def train_spa(data, spa, iterations):
    global INCLUDE_PREV_CONTEXT
    logloss_per_label = [[] for _ in range(len(spa))]  # List of lists for log-losses per label index
    for i in range(iterations):
        for row in data.itertuples():
            seq = row[1]
            label = row[2]
            
            # Encode sequence
            encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
            seq_len = len(seq)
            # Compute log-loss for the respective label's SPA
            train_logloss = spa[label].train_on_block(encoded_seq, include_prev_context=INCLUDE_PREV_CONTEXT) / seq_len
            
            # Append the computed log-loss to the appropriate label's list
            logloss_per_label[label].append(train_logloss)
    return train_logloss

def test_seq (data, spa, start_percentage, end_percentage, step):
    # for every test seq,
    # run it through all spas
    # classification = label associated with lowest loss spa
    # check classification against ground truth
    # compute accuracy (of all test runs)
    best_accuracy = 0
    best_context_percent = None
    for context_percent in range(start_percentage,end_percentage,step):
        print(f"Starting {context_percent}\n", flush=True)
        nb_correct = 0
        nb_test_total = 0
        for row in data.itertuples():
            seq = row[1]
            correct_label = row[2]
            encoded_seq = Sequence(seq, charmap=CharacterMap("ACGT"))
            seq_len = len(seq)
            nb_test_total += 1
            spa_logloss = []
            for index in range(len(spa)):
                #min_context = int(spa[index].get_tree_depth() * (context_percent / 100))
                time_start_backshift = time.perf_counter()
                spa_logloss.append(spa[index].compute_test_loss_backshift(encoded_seq, include_prev_context=False,min_context= 0) / seq_len)
                time_end_backshift = time.perf_counter()
                print("Backshifting", time_end_backshift-time_start_backshift)
                time_start_noback = time.perf_counter()
                spa_logloss.append(spa[index].compute_test_loss(encoded_seq, include_prev_context=False) / seq_len)
                time_end_noback = time.perf_counter()
                print("No backshift:",time_end_noback-time_start_noback)

            
            predicted_label = spa_logloss.index(min(spa_logloss))
            
            if predicted_label == correct_label:
                nb_correct += 1 
        
        accuracy = nb_correct / nb_test_total 
        print(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_context_percent = context_percent
    return best_accuracy, best_context_percent

def process_sequence(sequence, setting="remove", n=10):
    if setting == "remove":
        # Remove all characters that are not A, C, G, or T
        return ["".join(char for char in sequence if char in "ACGT")]
    
    elif setting == "random":
        # Replace each character that is not A, C, G, or T with a random nucleotide (A, C, G, or T)
        return ["".join(random.choice("ACGT") if char not in "ACGT" else char for char in sequence)]
    
    elif setting == "expand":
        # Generate 'n' sequences by replacing non-ACGT characters with random nucleotides
        expanded_sequences = []
        for _ in range(n):
            new_sequence = "".join(random.choice("ACGT") if char not in "ACGT" else char for char in sequence)
            expanded_sequences.append(new_sequence)
        return expanded_sequences
    else:
        raise ValueError("Setting must be 'remove', 'random', or 'expand'.")

def handle_N(data, setting="remove"):
    new_data = []
    for _, row in data.iterrows():
        sequence, label = row['sequence'], row['label']
        processed_sequences = process_sequence(sequence, setting)
        
        for proc_seq in processed_sequences:
            new_data.append({"sequence": proc_seq, "label": label})

    return pd.DataFrame(new_data)



# @profile
def main(dataset_folder, pretrain_file):
    global INCLUDE_PREV_CONTEXT
    global GAMMA
    global NB_TRAIN_ITERATIONS 
    global HANDLE_N_SETTING 
    global RATIO_PRETRAIN_TRAIN 
    
    global include_prev_contexts
    global gammas 
    global nb_train_iterations 
    global handle_N_settings 
    global ratio_pretrain_train 

    read_data_in_time = time.perf_counter()
    
    # read train, val, test data 
    train_path = f"{dataset_folder}/train.csv"
    val_path = f"{dataset_folder}/dev.csv"
    test_path = f"{dataset_folder}/test.csv"
    

    train_data = pd.read_csv(train_path)
    validation_data = pd.read_csv(val_path)
    
    ALPHABET_SIZE = 4
    unique_labels = train_data['label'].unique()
    
    with open(pretrain_file, 'r') as file:
        pretrain_data = file.read()
    
    # Train all SPAs using all possible combinations of hyperparams
    # Test all on validation set, return best SPA
    results_df = pd.DataFrame(columns=[
    "INCLUDE_PREV_CONTEXT", "GAMMA", "NB_TRAIN_ITERATIONS", 
    "HANDLE_N_SETTING", "RATIO_PRETRAIN_TRAIN", "VALIDATION ACCURACY"
    ])

    print("-----TRAINING")
    print("---SEARCH FOR BEST SPA(s)")
    print("nb_iterations , gamma, include_prev_context, handle_N_setting, ratio, minimum context, accuracy", flush=True)
    
    train_start_time = time.perf_counter()
    for include_prev_context, gamma, nb_iterations, handle_N_setting, ratio in itertools.product(
    include_prev_contexts, gammas, nb_train_iterations, handle_N_settings, ratio_pretrain_train
    ):  
        INCLUDE_PREV_CONTEXT = include_prev_context
        GAMMA = gamma
        NB_TRAIN_ITERATIONS = nb_iterations
        HANDLE_N_SETTING = handle_N_setting
        RATIO_PRETRAIN_TRAIN = ratio 
        

        train_data = handle_N(train_data, setting=HANDLE_N_SETTING)
        nb_train_seqs = len(train_data)
        seq_len = len(train_data.iloc[0, 0])
        nb_train_symbols = nb_train_seqs * seq_len
        
        # create list of spas based on number of labels: (spa_0 and spa_1 for labels 0, 1)
        spa = [LZ78SPA(ALPHABET_SIZE, gamma= GAMMA) for _ in unique_labels]


        nb_pretrain_symbols = math.ceil(RATIO_PRETRAIN_TRAIN * nb_train_symbols)
        pretrain_spa(pretrain_data, spa, nb_pretrain_symbols)
        spa_logloss = train_spa(train_data, spa, iterations= NB_TRAIN_ITERATIONS)
        # Test on validation test to assess this combination of hyperparams

        validation_data = handle_N(validation_data)
        accuracy, min_context = test_seq(validation_data, spa, 1, 2, 1)
        
        # print(nb_iterations, ", ", gamma,  ", ",include_prev_context,  ", ",handle_N_setting,  ", ",ratio,  ", ", accuracy, flush=True)
        print(f"{nb_iterations}, {gamma}, {include_prev_context}, {handle_N_setting}, {ratio}, {min_context}, {(accuracy * 100):.2f}", flush=True)

        
        
        current_result = pd.DataFrame([{
        "INCLUDE_PREV_CONTEXT": INCLUDE_PREV_CONTEXT,
        "GAMMA": GAMMA,
        "NB_TRAIN_ITERATIONS": NB_TRAIN_ITERATIONS,
        "HANDLE_N_SETTING": HANDLE_N_SETTING,
        "RATIO_PRETRAIN_TRAIN": RATIO_PRETRAIN_TRAIN,
        "MINIMUM_CONTEXT_PERCENTAGE": min_context,
        "VALIDATION ACCURACY": accuracy
        }])

        # Concatenate the current result with results_df
        results_df = pd.concat([results_df, current_result], ignore_index=True)

    
    # Find the best hyperparameter combination based on the highest accuracy
    print("---BEST SPA(s) FOUND")
    best_row = results_df.loc[results_df['VALIDATION ACCURACY'].idxmax()]
    best_params = best_row.to_dict()
    print("Best hyperparameters:", best_params)

    # Retrain and test using the best hyperparameters
    INCLUDE_PREV_CONTEXT = best_params["INCLUDE_PREV_CONTEXT"]
    GAMMA = best_params["GAMMA"]
    NB_TRAIN_ITERATIONS = int(best_params["NB_TRAIN_ITERATIONS"])
    HANDLE_N_SETTING = best_params["HANDLE_N_SETTING"]
    RATIO_PRETRAIN_TRAIN = best_params["RATIO_PRETRAIN_TRAIN"]
    best_min_context = best_params["MINIMUM_CONTEXT_PERCENTAGE"]

    # Retrain our best SPAs and use that to test on test data 
    spa = [LZ78SPA(ALPHABET_SIZE, gamma=GAMMA) for _ in unique_labels]
    train_data = handle_N(train_data, setting=HANDLE_N_SETTING)
    nb_train_seqs = len(train_data)
    seq_len = len(train_data.iloc[0, 0])
    nb_train_symbols = nb_train_seqs * seq_len
    nb_pretrain_symbols = math.ceil(RATIO_PRETRAIN_TRAIN * nb_train_symbols)

    pretrain_spa(pretrain_data, spa, nb_pretrain_symbols)
    spa_logloss = train_spa(train_data, spa, iterations=NB_TRAIN_ITERATIONS)



    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time

    
    
    # Final test
    print("-----TESTING")
    read_test_data_start_time = time.perf_counter()
    test_data = pd.read_csv(test_path)

    inference_start_time = time.perf_counter()
    # test_start_mem = memory_profiler.memory_usage()[0]

    test_data = handle_N(test_data)
    test_accuracy, trash_min_context = test_seq(test_data, spa, 1, 2, 1)
    #test_accuracy, trash_min_context = test_seq(test_data, spa, int(best_min_context), int(best_min_context)+1, 1)

    inference_end_time = time.perf_counter()
    # test_end_mem = memory_profiler.memory_usage()[0]
    print(f"Final accuracy with best hyperparameters: {(test_accuracy*100):.2f}")
    
        
    inference_duration = inference_end_time - inference_start_time

    print("Memory + tree depth of each spa")
    # for sp in spa:
    #     print(len(sp.to_bytes))
    
    print("labels", unique_labels)
    for sp in spa:
        spa_bytes = bytearray(sp.to_bytes())
        print(f"Mem in MB: {len(spa_bytes) / (1024 * 1024):.2f}", flush=True)
        makedirs("best_spas", exist_ok=True)

        # Extract the part after 'GUE/' and replace slashes with underscores
        binary_file_name = dataset_folder.split("GUE/", 1)[-1].replace("/", "_")

        # Create the full path for the binary file
        binary_file_path = os.path.join("best_spas", f"{binary_file_name}.bin")

        # Save the binary file
        with open(binary_file_path, 'wb') as file:
            file.write(spa_bytes)

        print("Tree depth", flush=True)
        sp.get_tree_depth()

        # sp.set_gamma(10)
    

    # print("-----TIME PROFILING+")
    # print(f"Read train + val data time: {(train_start_time - read_data_in_time): .5f}")
    # print(f"Number of training symbols: {nb_train_symbols}")
    # print(f"Length of one training sequence: {len(train_data.iloc[0, 0])}")
    # print(f"Total training time: {train_duration:.3f} seconds")
    

    # print(f"Number of test sequences: {len(test_data)}")
    # print(f"Length of test sequence: {len(test_data.iloc[0, 0])}")
    # print(f"Read test data time: {(inference_start_time - read_test_data_start_time): .5f}")
    # print(f"Total inference time: {inference_duration:.3f} seconds")
    # print(f"Inference time/symbol: {inference_duration/(len(test_data) * len(test_data.iloc[0, 0]))} seconds")

    # print("-----MEMORY REPORT")

    



if __name__ == "__main__":


    # parser = argparse.ArgumentParser(description="Script for training and testing SPA model")
    # parser.add_argument("-dataset_folder", type=str, required=True, help="Path to the folder containing train.csv and test.csv")
    # parser.add_argument("-pretrain_file", type=str, required=True, help="Path to the pretraining file")
    # args = parser.parse_args()



    
    parser = argparse.ArgumentParser(description="Script for training and testing SPA model")

    parser.add_argument("-dataset_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("-pretrain_file", type=str, required=True, help="Path to the pretraining file")
    parser.add_argument("--include_prev_context", type=str, required=True,
                        help="Set of values for INCLUDE_PREV_CONTEXT, e.g., '{True, False}'")
    parser.add_argument("--gamma", type=str, required=True,
                        help="Set of values for GAMMA, e.g., '{0.1, 0.5, 1.0}'")
    parser.add_argument("--nb_train_iterations", type=str, required=True,
                        help="Set of values for NB_TRAIN_ITERATIONS, e.g., '{1, 3, 5}'")
    parser.add_argument("--handle_n_setting", type=str, required=False,
                        help="Set of values for HANDLE_N_SETTING, e.g., '{remove, expand}'")
    parser.add_argument("--ratio_pretrain_train", type=str, required=True,
                        help="Set of values for RATIO_PRETRAIN_TRAIN, e.g., '{0.0, 0.1, 0.25}'")

    args = parser.parse_args()

    # Convert string inputs to Python sets
    include_prev_contexts = parse_bool(args.include_prev_context)
    

    gammas = parse_set(args.gamma)

    nb_train_iterations = parse_set(args.nb_train_iterations)
    nb_train_iterations = {int(x) for x in nb_train_iterations}

    handle_N_settings = {"remove"}

    ratio_pretrain_train = parse_set(args.ratio_pretrain_train)

    main(args.dataset_folder, args.pretrain_file)

    