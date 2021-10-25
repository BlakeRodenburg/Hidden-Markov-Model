import random
from decimal import *
import sys



def get_premade_input():

    """input format:
    states_int (n), vocab_int (m)
    m lines of strings for vocab words
    n lines of n decimals for adjacency table of transition probabilities
    n lines of m decimals for adjacency table of observation probabilities
    1 line 2 n initial probabilities
    1 line of x observations"""

def generate_model(states, vocab):

    transition_probability = [[0 for i in range(states)] for i in range(states)]
    observation_probability = [[0 for i in range(len(vocab))] for i in range(states)]
    initial_probability = [0 for i in range(states)]

    for i in range(states):
        remaining_p = 1
        for j in range(states - 1):
            next_p = round(random.uniform(0.0000000001, 0.8 * remaining_p), 2)
            transition_probability[i][j] = next_p
            remaining_p -= next_p
        transition_probability[i][-1] = round(remaining_p, 2)

    for i in range(states):
        remaining_p = 1
        for j in range(len(vocab) - 1):
            next_p = round(random.uniform(0.0000000001, 0.8 * remaining_p), 2)
            observation_probability[i][j] = next_p
            remaining_p -= next_p
        observation_probability[i][-1] = round(remaining_p, 2)

    remaining_p = 1
    for i in range(states - 1):
        next_p = round(random.uniform(0.0000000001, 0.8 * remaining_p), 2)
        initial_probability[i] = next_p
        remaining_p -= next_p
    initial_probability[-1] = round(remaining_p, 2)

    return transition_probability, observation_probability, initial_probability

def generate_sequences(transition_probability, observation_probability, initial_probability, sequence_length, vocab):

    sequence_list = []
    states = len(transition_probability)

    for i in range(sequences): # for each sequence

        next_state_p = round(random.uniform(0, 1), 2)
        for j in range(states): # for each possible start state within a sequence
            if initial_probability[j] >= next_state_p or j == states - 1:
                current_state = j
                next_observation_p = round(random.uniform(0, 1), 2)
                for k in range(len(observation_probability[j])): # for each observation within the chosen start state
                    if observation_probability[j][k] >= next_observation_p or k == len(observation_probability[j]) - 1:
                        sequence_list.append(f"{vocab[k]}")
                        break
                    else:
                        next_observation_p -= observation_probability[j][k]
                break
            else:
                next_state_p -= initial_probability[j]

        for l in range(sequence_length - 1):
            next_state_p = round(random.uniform(0, 1), 2)
            for m in range(states): # for each possible start state within a sequence
                if transition_probability[current_state][m] >= next_state_p or m == states - 1:
                    next_observation_p = round(random.uniform(0, 1), 2)
                    for n in range(len(observation_probability[m])): # for each observation within the chosen start state
                        if observation_probability[m][n] >= next_observation_p or n == len(observation_probability[m]) - 1:
                            sequence_list[i] += f" {vocab[n]}"
                            break
                        else:
                            next_observation_p -= observation_probability[m][n]
                    current_state = m
                    break
                else:
                    next_state_p -= transition_probability[current_state][m]


    return sequence_list

def print_model(transition_probability, observation_probability, initial_probability, sequence_list):

    states = len(transition_probability)


    for i in range(states):
        print_string = ""
        for j in range(states):
            print_string += f" {transition_probability[i][j]}"
        print(print_string[1:])

    for i in range(states):
        print_string = ""
        for j in range(len(observation_probability[0])):
            print_string += f" {observation_probability[i][j]}"
        print(print_string[1:])

    print_string = ""
    for i in range(states):
        print_string += f" {initial_probability[i]}"
    print(print_string[1:])

    for sequence in sequence_list:
        print(sequence)

    return

if __name__ == "__main__":

    """
    
    
    """

    print("Enter number of sequences output from model:")
    sequences = int(sys.stdin.readline().strip())
    print("Enter sequence length:")
    sequence_length = int(sys.stdin.readline().strip())

    states = 2
    vocab = {0: 0, 1: 1, 2: 2}

    print(states, "3", sequences)
    for i in range(len(vocab)):
        print(vocab[i])

    transition_probability, observation_probability, initial_probability = generate_model(states, vocab)
    sequence_list = generate_sequences(transition_probability, observation_probability, initial_probability, sequence_length,  vocab)
    print_model(transition_probability, observation_probability, initial_probability, sequence_list)

