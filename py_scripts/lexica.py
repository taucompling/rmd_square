# generation of lexica and prior
import numpy as np
from itertools import product, combinations, combinations_with_replacement


def get_lexica(s_amount, m_amount, target_lex, competitor_lex, mutual_exclusivity=True):
    """creates lexica

    :param s_amount: number of columns in lexicon
    :type s_amount: int
    :param m_amount: number of lines in lexicon
    :type m_amount: int
    :param target_lex: lexicon of target type
    :type: list
    :param competitor_lex: lexicon of competitor type
    :type: list
    :param mutual_exclusivity: no concept assigned to more than one message, defaults to True
    :type mutual_exclusivity: bool, optional
    :return: list with transposed lexica (np.arrays): states = rows, messages = columns, list with target type, list with competitor type
    :rtype: list, list, list
    """
    target_index, competitor_index = [], []
    columns = list(product([0., 1.], repeat=s_amount)) 
    columns.remove(tuple(np.zeros(s_amount)))  # remove message false of all states
    columns.remove(tuple(np.ones(s_amount)))  # remove message true of all states
    if mutual_exclusivity:
        # no concept assigned to more than one message
        matrix = list(combinations(columns, r=m_amount)) #combinations = not repeated element
        out = []
        for typ, mrx in enumerate(matrix):
            lex = np.array([mrx[i] for i in range(m_amount)]) # converts tuples in matrix to np.arrays (war mal s_amount)
            lex = np.transpose(np.array([mrx[i] for i in range(m_amount)])) # transposed # war mal s_amount
            if np.array_equal(lex, np.array(target_lex)):
                target_index.append(typ)
            if np.array_equal(lex, np.array(competitor_lex)):
                competitor_index.append(typ)
            out.append(lex)
    else:
        # If we allow for symmetric lexica (=repeated elements)
        matrix = list(product(columns, repeat=m_amount))
        # matrix = list(combinations_with_replacement(columns,m_amount)) #only 112 types
        out = []
        for typ, mrx in enumerate(matrix):
            lex = np.array([mrx[i] for i in range(m_amount)]) # war mal in range s_amount
            lex = np.transpose(np.array([mrx[i] for i in range(m_amount)])) # war mal s_amount
            if np.array_equal(lex, np.array(target_lex)):
                target_index.append(typ)
            if np.array_equal(lex, np.array(competitor_lex)):
                competitor_index.append(typ)
            out.append(lex)
    return out, target_index, competitor_index # target_index and competitor_index may be empty


def get_lexica_bins(lexica_list, all_states):
    """get the concept of lexica

    :param lexica_list: list with lexica: states = rows, messages = columns
    :type lexica_list: list
    :return: list with lexica's concepts
    :rtype: list
    """

    concepts = []
    for state in all_states:
        concepts += list(product([0,1], repeat=state))
    lexica_concepts = [] # contains for each lexicon a list with concepts
    for lex in lexica_list:
        concept_indices = []
        current_lex = np.transpose(lex) 
        for concept in current_lex:
            if np.array_equal(concept, np.array([-100, -100, -100])):
                continue
            concept_indices.append(concepts.index(tuple(concept)))
        lexica_concepts.append(concept_indices)

    bin_counter = [] # contains lexica_concepts (sorted)
    bins = [] # contains lists with index of lexica containing the same concept
    for lex_idx in range(len(lexica_list)):
        sorted_lexica_concepts = lexica_concepts[lex_idx] # get lexica_concept for current dict
        sorted_lexica_concepts.sort()
        if not (sorted_lexica_concepts in bin_counter):
            bin_counter.append(sorted_lexica_concepts)
            bins.append([lex_idx])
        else:
            bins[bin_counter.index(sorted_lexica_concepts)].append(lex_idx)

    # up to here we get bins for a single linguistic behavior. Now we double that for Literal/Gricean split
    gricean_bins = []
    for b in bins:
        g_bin = [x+len(lexica_list) for x in b] # pragmatic dicts have literal value + len(lexicon)
        gricean_bins.append(g_bin)
    bins = bins+gricean_bins
    return bins

def normalize_cost(cost_dict, all_states):
    """
    cost = max(cost) - cost +1 and then normalize it
    :param cost_dict: dictionary with costs for messages
    :type cost_dict: dict
    :param all_states: list of states
    :type: list
    :return: dictionary with normalized costs
    :rtype: dictionary
    """
    k_length_cost = {}
    for key, value in cost_dict.items():
        try:
            k_length_cost[len(key)].append(value)
        except KeyError:
            k_length_cost[len(key)] = [value]

    for state in all_states:
        if state not in k_length_cost:
            raise Exception(f"No cost function defined for {state} states. Add it in lexica.py in get_prior().")

    for key, value in cost_dict.items():
        cost_dict[key] = (max(k_length_cost[len(key)]) - value + 1)/np.sum(k_length_cost[len(key)])

    return cost_dict


def get_prior(lexica_list, cost, all_states):
    """Calculates prior over lexicon

    :param lexica_list: list of lexica
    :type lexica_list: list of lists
    :return: list of priors for each lexicon
    :rtype: list
    """

    concepts = []
    for state in all_states:
        concepts += list(product([0,1], repeat=state))
        concepts.remove(tuple(np.zeros(state)))  # remove message false of all states
        concepts.remove(tuple(np.ones(state)))  # remove message true of all states
    if cost == "brochhagen": # cost of each concept in 'concepts'
        cost_dict = {(0,0,1): 3, (0,1,0):8, (0,1,1):4, (1,0,0):4, (1,0,1):10, (1,1,0):5, # for three states
                    (0,1):3, (1,0):4}  # for two states
    elif cost == "building_blocks":
        cost_dict = {(0,0,1): 8, (0,1,0): 21, (0,1,1): 8, (1,0,0): 7, (1,0,1): 18, (1,1,0): 10, # for three states
                    (0,1):8, (1,0):7}  # for two states
    else:
        raise Exception("Cost function is not defined. Please define it in lexica.py")


    cost_dict = normalize_cost(cost_dict, all_states)
    out = []
    for lex in lexica_list:
        current_lex = np.transpose(lex)
        lex_val = 1  # probability of current lexicon's concepts
        for concept in current_lex:

            if np.array_equal(concept, np.array([-100, -100, -100])):
                continue

            lex_val *= cost_dict[tuple(concept)]
        out.append(lex_val)
    out = out + out  # double for two types of linguistic behavior
    return np.array(out) / np.sum(out)
