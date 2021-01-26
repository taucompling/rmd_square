import numpy as np
from itertools import product, combinations, combinations_with_replacement, permutations
from py_scripts.lexica import get_indices, pad_lex

def get_predefined_lexica(path, target_lex, competitor_lex):
    """Converts lexica from txt.document to numpy arrays and pads them with -5 if necessary

    :param path: path to lexica
    :type path: string
    :param target_lex: lexicon of target type
    :type: list
    :param competitor_lex: lexicon of competitor type
    :type: list
    :return: list with lexica (np.arrays): states = rows, messages = columns, list with target type, list with competitor type, all_states, all_messages
    :rtype: list, list, list, list, list
    """
    with open(path, "r") as input_lex:
        lexica =  [np.array(eval(lex)) for lex in input_lex]
    target_index, competitor_index = [], [] 
    all_messages = set()
    all_states = set()
    for l in lexica: 
        s = np.array(l).shape[0]
        m = np.array(l).shape[1]
        all_messages.add(m)
        all_states.add(s)

    all_messages = list(all_messages)
    states = list(all_states)[0]
    max_message = max(all_messages)

    out = []
    for lex in lexica:
        if np.array_equal(lex, np.array(target_lex)):
            target_index.append(len(out))
        if np.array_equal(lex, np.array(competitor_lex)):
            competitor_index.append(len(out))
        lex = [lex]
        if max_message != np.array(lex[0]).shape[1]:
            indices_messages = get_indices(max_message, np.array(lex[0]).shape[1])
            lex = pad_lex(indices_messages, lex[0], "column")

        out += lex

    return out, target_index, competitor_index, states, all_messages