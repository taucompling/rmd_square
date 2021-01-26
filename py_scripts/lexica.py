# generation of lexica and prior
import numpy as np
from itertools import product, combinations, combinations_with_replacement, permutations

def get_indices(max_message, m_amount):
    per = set(permutations([0 for x in range(m_amount)] + [1 for x in range(max_message - m_amount)]))
    indices = []
    for p in per:
        ix = np.where(np.array(p)==1)
        indices.append(ix)
    return indices

def pad_lex(indices, lex, row_column):
    padded_lex = []
    for idx in indices:
        new_lex = lex
        for i in idx[0]:
            if row_column == "column":
                new_lex = np.insert(new_lex, i, values=-5, axis=1)
            else:
                new_lex = np.insert(new_lex, i, values=-5, axis=0)
                
        padded_lex.append(new_lex)
    return padded_lex

def get_lexica(s_amount, m_amount, max_message, target_lex, competitor_lex, mutual_exclusivity=True, puzzle=False):
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
    :return: list with lexica (np.arrays): states = rows, messages = columns, list with target type, list with competitor type
    :rtype: list, list, list
    """
    target_index, competitor_index = [], []
    columns = list(product([0., 1.], repeat=s_amount)) 
    columns.remove(tuple(np.zeros(s_amount)))  # remove message false of all states
    columns.remove(tuple(np.ones(s_amount)))  # remove message true of all states

    if puzzle and s_amount == 3:
        columns.remove((1, 0, 1))
        columns.remove((0, 1, 0))

    if max_message != m_amount:
        indices_messages = get_indices(max_message, m_amount)
    if mutual_exclusivity:  # no concept assigned to more than one message
        matrix = list(combinations(columns, r=m_amount)) #combinations = not repeated element
    else: # If we allow for symmetric lexica (=repeated elements)
        matrix = list(product(columns, repeat=m_amount))
    out = []
    for typ, mrx in enumerate(matrix):
        lex = np.array([mrx[i] for i in range(m_amount)]) 
        lex = [np.transpose(np.array([mrx[i] for i in range(m_amount)]))]
        if np.array_equal(lex[0], np.array(target_lex)):
            target_index.append(len(out))
        if np.array_equal(lex[0], np.array(competitor_lex)):
            competitor_index.append(len(out))
        if max_message != m_amount:
            lex = pad_lex(indices_messages, lex[0], "column")
                
        out += lex

    return out, target_index, competitor_index # target_index and competitor_index may be empty


def get_lexica_bins(lexica_list, states, puzzle):
    """get the concept of lexica

    :param lexica_list: list with lexica: states = rows, messages = columns
    :type lexica_list: list
    :return: list with lexica's concepts
    :rtype: list
    """

    concepts = list(product([0,1], repeat=states))
    lexica_concepts = [] # contains for each lexicon a list with concepts
    for lex in lexica_list:
        concept_indices = []
        current_lex = np.transpose(lex) 
        for concept in current_lex:
            if -5 in concept:
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

    if not puzzle:
        gricean_bins = []
        for b in bins:
            g_bin = [x+len(lexica_list) for x in b] # pragmatic dicts have literal value + len(lexicon)
            gricean_bins.append(g_bin)
        bins = bins+gricean_bins

    return bins

def get_prior(lexica_list, puzzle, cost_dict):
    """Calculates prior over lexicon

    :param lexica_list: list of lexica
    :type lexica_list: list of lists
    :return: list of priors for each lexicon
    :rtype: list
    """
  
    out = []
    for lex in lexica_list:
        current_lex = np.transpose(lex)
        lex_val = 1  # probability of current lexicon's concepts
        for concept in current_lex:
            if -5 in concept:
                continue
            lex_val *= cost_dict[tuple(concept)]
        out.append(lex_val)
    if not puzzle:
        out = out + out  # double for two types of linguistic behavior
    return np.array(out) / np.sum(out)
