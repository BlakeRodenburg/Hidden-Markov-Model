import random
from decimal import *
import copy
import sys
import math


def get_premade_input():
    """input format:
    states_int (n), vocab_int (m), sequence_count (k)
    m lines of strings for vocab words
    n lines of n decimals for adjacency table of transition probabilities
    n lines of m decimals for adjacency table of observation probabilities
    1 line of n initial probabilities
    k line of x observations"""

    states_int, vocab_int, sequence_count = map(int, sys.stdin.readline().split())

    vocab = {}
    state_transitions = [[0 for i in range(states_int)] for i in range(states_int)]
    observation_probabilities = [[0 for i in range(vocab_int)] for i in range(states_int)]
    observations = []

    for i in range(vocab_int):
        # creates vocab dictionary where the value of the key is the corresponding observation index
        vocab[sys.stdin.readline().strip()] = i

    for i in range(states_int):  # fills out the state transition table
        current_transition_probabilities = list(sys.stdin.readline().split())
        for j in range(states_int):
            state_transitions[i][j] = Decimal(current_transition_probabilities[j])

    for i in range(states_int):  # fills out the observation probability table
        current_observation_probabilities = list(sys.stdin.readline().split())
        for j in range(vocab_int):
            observation_probabilities[i][j] = Decimal(current_observation_probabilities[j])

    initial_probabilities = list(sys.stdin.readline().split())
    # creates a list of start state probabilities with index corresponding to state

    for i in range(sequence_count):
        observations.append(list(sys.stdin.readline().split()))
    # Creates a list of observation sequences composed of words in the vocab dictionary

    return state_transitions, observation_probabilities, initial_probabilities, vocab, observations


def get_forward(state_transitions, observation_probability, vocab, observations, initial_probabilities):
    """ This function returns the table of probabilities for each observation path where each memory location
    contains the summed probability for every possible state path for a given observation path starting from the
     first observation"""

    number_of_states = len(state_transitions)
    probability_paths = [[Decimal(0) for i in range(len(observations))] for i in range(number_of_states)]

    for i in range(number_of_states):  # first probabilities use the initial probability of starting in a state.
        probability_paths[i][0] = round(
            Decimal(initial_probabilities[i]) * observation_probability[i][vocab[observations[0]]], 2)
        # remove round in final product, just for test case

    for i in range(1, len(observations)):  # for each point in time
        for j in range(number_of_states):  # for each state
            sum_of_probabilities = Decimal(0)  # resets sum_of_prbability
            for k in range(number_of_states):  # for each past state
                current_probability = probability_paths[k][i - 1] * observation_probability[j][vocab[observations[i]]] * \
                                      state_transitions[k][j]
                # calculates the probability for observation path i-1 ending in state k
                # going to observation path i ending in state j
                sum_of_probabilities += current_probability
                # sums probability of all observation paths j -1 moving to observation path i in state j

            probability_paths[j][i] = sum_of_probabilities

    return probability_paths


def get_backward(state_transitions, observation_probability, vocab, observations):
    """ This function returns the table of probabilities for each reversed observation path where each memory location
    contains the summed probability for every possible state path for a given observation path starting from the
     last observation with a probability of 1 and working backwards"""

    number_of_states = len(state_transitions)
    probability_paths = [[Decimal(0) for i in range(len(observations))] for i in range(number_of_states)]

    for i in range(number_of_states):
        probability_paths[i][-1] = Decimal(1)  # * observation_probability[i][vocab[observations[-1]]]
        # no way of start state prob, therefore all states have an equal chance.

    for i in range(len(observations) - 2, -1, -1):  # for each point in time
        for j in range(number_of_states):  # for each past state
            sum_of_probabilities = Decimal(0)
            for k in range(number_of_states):  # for each future state
                current_probability = state_transitions[j][k] * observation_probability[k][vocab[observations[i + 1]]] * \
                                      probability_paths[k][i + 1]
                # calculates the probability for observation path i +1 ending in state k
                # going to observation path i ending in state j
                sum_of_probabilities += current_probability
                # sums probability of all observation paths j + 1 moving to observation path i in state j
            probability_paths[j][i] = sum_of_probabilities

    return probability_paths


def get_gamma_table(forward_p, backward_p):
    """ Function creates the gamma table. The probability of the model being in a given state, at a given observation point.
     It is identified my multiplying the forward and backward probability of a state ate a given observation point against
      the probability of the model as a a whole"""

    full_model_p = Decimal(0)
    observations = len(forward_p[0])
    states = len(forward_p)
    gamma_table = [[Decimal(0) for i in range(observations)] for i in range(states)]

    for i in range(states):
        full_model_p += forward_p[i][-1]

    for t in range(observations):
        for j in range(states):
            gamma_table[j][t] = (forward_p[j][t] * backward_p[j][t]) / full_model_p

    return gamma_table


def get_xi_table(state_transitions, observation_probability, forward_p, backward_p, observations, vocab):
    """ Function creates the xi table. The probability of the model transitioning between one state to another and
     emitting a given observation at a given observation point. It is identified by multiplying the forward probability
     of the intitial state by the backwards probability of the next state, multiplying that with the transition
     probability between states, multiplying that total by the probability of the next state emitting the observation
      and dividing the product by the probability of the model as a whole"""

    full_model_p = Decimal(0)

    states = len(forward_p)
    xi_table = [[[Decimal(0) for i in range(states)] for i in range(states)] for i in range(len(observations) - 1)]

    for i in range(states):
        full_model_p += forward_p[i][-1]

    for t in range(len(observations) - 1):  # for each point in the obesrvation path
        for i in range(states):  # for each starting state
            for j in range(states):  # for each state the starting state can transition to
                xi_table[t][i][j] = (forward_p[i][t] * backward_p[j][t + 1] * state_transitions[i][j] *
                                     observation_probability[j][vocab[observations[t + 1]]]) / full_model_p
                # forward_p of i * backward_p of j * i to j transition_p * j observation[t + 1] output p

    return xi_table


def reestimate_state_transitions(forward_p, xi_table, observations, one_over_model_p):
    """ Function reestimates the state transition table. It does this by dividing the summed probability of every i to
    j state transition across all observations with the summed probability of every i state transition across an
     observation sequence. After all sequence specific reestimates are predicted the results are integrated. This is
      then normalised so that the sum of transition probabilities for a given state is 1 """

    states = len(forward_p[0])
    new_state_transitions = [[[Decimal(0) for i in range(states)] for i in range(states)] for i in
                             range(len(observations))]
    final_state_transitions = [[Decimal(0) for i in range(states)] for i in range(states)]
    i_k_sum = [[Decimal(0) for i in range(states)] for i in
               range(len(observations))]  # sum of transitions from i to any state
    # Xi table =  The probability of the model transitioning between one state to another and emitting a given
    # observation at a given observation point.

    for m in range(len(observations)):  # for each observation sequence
        current_xi_table = xi_table[m]  # makes code more readable

        for t in range(len(observations[0]) - 1):
            for i in range(states):
                for j in range(states):
                    i_j_transition_at_t_probability = current_xi_table[t][i][j]
                    i_k_sum[m][i] += i_j_transition_at_t_probability
                    new_state_transitions[m][i][
                        j] += i_j_transition_at_t_probability  # sum of transitions from state i to j.

    i_prob_total = [Decimal(0) for i in range(states)]

    for m in range(len(observations)):  # for each observation sequence to be integrated.
        for i in range(states):
            for j in range(
                    states):  # Divides the sum of a specific transition for a state by the sum of all transitions for a state.
                final_state_transitions[i][j] += new_state_transitions[m][i][j] / i_k_sum[m][j]
                i_prob_total[i] += new_state_transitions[m][i][j] / i_k_sum[m][
                    j]  # sum of transition probabilities for i, used in normalisation
                # Potentially broken, if final algorithm is bugged check over. Without the one over model it matches Baum Welch with 1 sequence.

    for i in range(states):  # normalises probabilities so all transition probabilities for a state sum to 1.
        normalisation_factor = 1 / i_prob_total[i]

        for j in range(states):
            final_state_transitions[i][j] = Decimal(round(final_state_transitions[i][j] * normalisation_factor, 2))

    return final_state_transitions


def reestimate_observation_probability(gamma_table, observations, vocab, states):
    """ Function re-estimates the observation probability table for each state for each sequence. It does this by dividing the summed
     probability of observing a given observation in state i by the summed probability of all observation probabilities
     given state i. After all sequence specific reestimates are predicted the results are integrated. This is then
      normalised so that the sum of observartion probabilities for a given state is 1."""

    sum_of_state_obs_prob = [[Decimal(0) for i in range(states)] for i in range(len(observations))]
    new_observation_probability = [[[Decimal(0) for i in range(len(vocab))] for i in range(states)] for i in
                                   range(len(observations))]
    # gamme_table = The probability of the model being in a given state, at a given observation point.

    for m in range(len(observations)):  # for each observation sequence
        current_observation = observations[m]  # makes code more readable
        current_gamma_table = gamma_table[m]  # makes code more readable
        for j in range(states):
            for t in range(len(current_observation)):
                sum_of_state_obs_prob[m][j] += current_gamma_table[j][t]
                new_observation_probability[m][j][vocab[current_observation[t]]] += current_gamma_table[j][
                    t]  # sum of probabilities for
                # observation t from state j from gamma table

    i_prob_total = [Decimal(0) for i in range(states)]
    final_observation_probability = [[Decimal(0) for i in range(len(vocab))] for i in range(states)]

    for m in range(len(observations)):
        for i in range(states):
            for j in range(
                    len(vocab)):  # Divides the sum of a specific observation probability for a state by the sum of
                # all observation probabilities for a state.
                final_observation_probability[i][j] += new_observation_probability[m][i][j] / sum_of_state_obs_prob[m][
                    i]
                i_prob_total[i] += new_observation_probability[m][i][j] / sum_of_state_obs_prob[m][
                    i]  # sum of observation probabilities for i, used in
                # normalisation

    for i in range(states):  # normalises probabilities so all observation probabilities for a state sum to 1.
        normalisation_factor = 1 / i_prob_total[i]
        for j in range(len(vocab)):
            final_observation_probability[i][j] = Decimal(
                round(final_observation_probability[i][j] * normalisation_factor, 2))

    return final_observation_probability


def reestimate_initial_probabilites(gamma_table):
    """ Reestimates initial probabilities of stating in a given state for each sequence. It does this by taking the sum probability of
    being in a state at observation 0 and divides it by the total probability of all states.  After all sequence specific reestimates are predicted the results are integrated. This is then normaliesed so
    the total initial probabilities == 1. """

    initial_probabilities = [[Decimal(0) for i in range(len(gamma_table[0]))] for i in range(len(gamma_table))]
    total_start_p = [Decimal(0) for i in range(len(gamma_table))]

    for m in range(len(gamma_table)):
        current_gamma_table = gamma_table[m]
        for i in range(len(initial_probabilities[0])):  # for each state
            total_start_p[m] += current_gamma_table[i][0]
            initial_probabilities[m][i] = current_gamma_table[i][0]

    i_prob_total = Decimal(0)  # used in determining normalisation factor

    final_initial_probabilities = [Decimal(0) for i in range(len(gamma_table[0]))]
    for m in range(len(gamma_table)):
        for i in range(len(initial_probabilities[0])):  # finds unnormalised initiation probabilities.
            final_initial_probabilities[i] += initial_probabilities[m][i] / total_start_p[m]
            i_prob_total += initial_probabilities[m][i] / total_start_p[m]

    normalisation_factor = Decimal(1 / i_prob_total)
    for i in range(len(initial_probabilities[0])):  # applies normalisation.
        final_initial_probabilities[i] = final_initial_probabilities[i] * normalisation_factor

    return final_initial_probabilities


class HMM_Model:
    """ Class contains all model unique  data structures.

    """

    def __init__(self, state_transitions, observation_probability, initial_probabilities, sequence_count):
        self.state_transitions = state_transitions
        self.observation_probability = observation_probability
        self.initial_probabilities = initial_probabilities
        self.forward_p = [0 for i in range(sequence_count)]
        self.backward_p = [0 for i in range(sequence_count)]
        self.gamma_table = [0 for i in range(sequence_count)]
        self.xi_table = [0 for i in range(sequence_count)]
        self.new_state_transitions = []
        self.new_observation_probability = []
        self.new_initial_probabilities = []


def generate_random_model(states, vocab):
    """ Function creates an initial random model to try and fit the observation sequences. Multiple models fitted
    reduces the probability of finding an answer in a local peak, rather finding the global maximum likelihood.
    All probability rows sum to 1.

    """
    transition_probability = [[0 for i in range(states)] for i in range(states)]
    observation_probability = [[0 for i in range(vocab)] for i in range(states)]
    initial_probability = [0 for i in range(states)]

    for i in range(states):  # builds transtition table.
        remaining_p = 1
        for j in range(states - 1):
            next_p = round(random.uniform(0.0000000001, 0.8 * remaining_p), 2)
            transition_probability[i][j] = round(Decimal(next_p), 2)
            remaining_p -= next_p
        transition_probability[i][-1] = round(Decimal(remaining_p), 2)

    for i in range(states):  # builds emission table.
        remaining_p = 1
        for j in range(vocab - 1):
            next_p = round(random.uniform(0.0000000001, 0.8 * remaining_p), 2)
            observation_probability[i][j] = round(Decimal(next_p), 2)
            remaining_p -= next_p
        observation_probability[i][-1] = round(Decimal(remaining_p), 2)

    remaining_p = 1
    for i in range(states - 1):  # builds initial probability table.
        next_p = round(random.uniform(0.0000000001, 0.8 * remaining_p), 2)
        initial_probability[i] = next_p
        remaining_p -= next_p
    initial_probability[-1] = round(remaining_p, 2)

    return transition_probability, observation_probability, initial_probability


def get_random_model_probabilities(forward_p):
    """ Defunct function, built as part of the rabinar paper requirement for determining model
     contribution to reestimates."""
    observations = len(forward_p)

    model_p_list = [Decimal(0) for i in range(observations)]

    for i in range(observations):
        current_forward_p = forward_p[i]
        for j in range(len(current_forward_p)):
            model_p_list[i] += current_forward_p[j][-1]
        model_p_list[i] = Decimal(1) / model_p_list[i]

    return model_p_list


def model_probability(model):
    """ Finds the -log likelyhood of a model providing all the observation sequences input into the program.
     It does this by taking the -log of the product of the total likelyhood of each observation sequence."""

    observations = len(model.forward_p)  # number of observations.
    states = len(model.forward_p[0])  # number of states
    full_model_observation_probability = [Decimal(0) for i in
                                          range(observations)]  # list of model probabilities for each observation

    for i in range(observations):  # fills full_model_observation_probability
        for j in range(states):
            full_model_observation_probability[i] += model.forward_p[i][j][-1]

    full_model_p = 1

    for i in range(len(full_model_observation_probability)):
        full_model_p = full_model_p * full_model_observation_probability[i]

    full_model_p = -Decimal.ln(full_model_p)  # takes natural log of the final value.
    return Decimal(round(full_model_p, 2))


def output_result(model_list):
    """ Function outputs the """

    states = len(model_list[0].forward_p[0])

    print("Original Model")
    print("-log likelihood: " + str(model_probability(model_list[0])) + "   ")
    print("State transition probability:")

    for i in range(states):
        new_line = ""
        for j in range(states):
            new_line = new_line + str(model_list[0].state_transitions[i][j]) + "   "
        print(new_line)

    print("Observation Emission probability:")
    for i in range(states):
        new_line = ""
        for j in range(len(vocab)):
            new_line = new_line + str(model_list[0].observation_probability[i][j]) + "   "
        print(new_line)

    ranking_list = []
    for m in range(1, len(model_list)):
        ranking_list.append([model_probability(model_list[m]), m])
    ranking_list.sort()

    print()
    print("Ascending HMM results:")
    print()
    for m in range(len(ranking_list)):
        print("-log likelihood: " + str(model_probability(model_list[ranking_list[m][1]])))
        print("State transition probability:")

        for i in range(states):
            new_line = ""
            for j in range(states):
                new_line = new_line + str(model_list[ranking_list[m][1]].state_transitions[i][j]) + "   "
            print(new_line)

        print("Observation Emission probability:")
        for i in range(states):
            new_line = ""
            for j in range(len(vocab)):
                new_line = new_line + str(model_list[ranking_list[m][1]].observation_probability[i][j]) + "   "
            print(new_line)

        print()


if __name__ == "__main__":

    """ HMM algorithm.
    
    Notes: 
    Algorithm assumes all nodes are capable of being connected for simplicity in test case generation. This has consequences
    on its usefulness as the highly connected graphs it takes as inputs and their observations have a probability landscape 
    with numerous local maximi. Particularly if the test case generator has more than 2 states. Functionality could be 
    improved (and is needed for bioinformatic applications) so the an input of -1 signifies no edge between nodes and
    the algorithm would adjust reestimates accordingly. Despite this the Algorithm is consistently able to find models 
    more likely than the initial model input into the program and appears to work.
    
    On the same theme. Input models with extremely high or low probabilities are less likely to result in a similar 
    model due to the limited random model generation and the number of possible models with similar initial conditions.
    
    Resources used:
    
    https://web.stanford.edu/~jurafsky/slp3/A.pdf
    https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Example
    https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf
    
     
    

            """

    # model = state_transitions, observation_probability, vocab, observations, initial_probabilities
    state_transitions, observation_probability, initial_probabilities, vocab, observations = get_premade_input()
    true_model = HMM_Model(state_transitions, observation_probability, initial_probabilities, len(observations))

    model_list = [true_model]

    for k in range(len(observations)):
        model_list[0].forward_p[k] = get_forward(model_list[0].state_transitions,
                                                 model_list[0].observation_probability, vocab, observations[k],
                                                 model_list[0].initial_probabilities)



    for i in range(20):
        random_model = HMM_Model(*generate_random_model(len(state_transitions), len(vocab)), len(observations))
        model_list.append(random_model)

    for i in range(100):  # Loop until answer found

        for j in range(1, len(model_list)):  # For each model
            current_model = model_list[j]

            for k in range(len(observations)):  # For each observation sequence

                # Estimate Step
                current_model.forward_p[k] = get_forward(current_model.state_transitions,
                                                         current_model.observation_probability, vocab, observations[k],
                                                         current_model.initial_probabilities)
                current_model.backward_p[k] = get_backward(current_model.state_transitions,
                                                           current_model.observation_probability, vocab,
                                                           observations[k])

                current_model.gamma_table[k] = get_gamma_table(current_model.forward_p[k], current_model.backward_p[k])
                current_model.xi_table[k] = get_xi_table(current_model.state_transitions,
                                                         current_model.observation_probability,
                                                         current_model.forward_p[k], current_model.backward_p[k],
                                                         observations[k], vocab)

            # Maximisation step
            one_over_model_p = get_random_model_probabilities(current_model.forward_p)
            current_model.new_state_transitions = reestimate_state_transitions(current_model.forward_p,
                                                                               current_model.xi_table, observations,
                                                                               one_over_model_p)
            current_model.new_observation_probability = reestimate_observation_probability(current_model.gamma_table,
                                                                                           observations, vocab,
                                                                                           len(state_transitions))
            current_model.new_initial_probabilities = reestimate_initial_probabilites(current_model.gamma_table)

            current_model.state_transitions = current_model.new_state_transitions
            current_model.initial_probabilities = current_model.new_initial_probabilities
            current_model.observation_probability = current_model.new_observation_probability  # Disabled for adeveloperdiary testing

    output_result(model_list)
