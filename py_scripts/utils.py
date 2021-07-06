import numpy as np
from fractions import Fraction

def print_end_results(bin_orders, bin_winner, runs, lexica, puzzle, sum_winning_types, print_x):
    end_results = [
    "-------------------------------------------------------------------------------------------------------------------------------------------------\n",     
    f"# WINNER BIN in {round(bin_orders[bin_winner][1]/runs * 100,2)} percent of the runs:\n",
    f"# Average Proportion: {round(bin_orders[bin_winner][0]/bin_orders[bin_winner][1],4)}\n",
    f"{get_lexica_representations(bin_winner[0], lexica, puzzle)}\n",    
    "-------------------------------------------------------------------------------------------------------------------------------------------------\n",    
    f"# Summed proportion of the bin of the incumbent: {round(sum_winning_types,4)}\n",
    "-------------------------------------------------------------------------------------------------------------------------------------------------\n",    
    f"# {print_x if print_x > 0 else 10} best types: \n"]

    return end_results

def get_type_bin(type_idx, bins):

    for b in range(len(bins)):
        if type_idx in bins[b]:
            return b
    raise Exception("Something did not work out with the bins!")


def get_lexica_representations(type, lexica, puzzle):
    l = lexica[type-len(lexica)]
    for column in l.T:
        if -5 in column:
            l = np.delete(l, np.where(l == -5)[1][0], 1)
    return f"{check_literal_or_pragmatic(type, lexica, puzzle)}:\n{l}"


def check_literal_or_pragmatic(type, lexica, puzzle):
    if type > len(lexica)-1 or puzzle:
        return "pragmatic"
    return "literal"


def get_target_bins(type_idx, bins, level, number_of_lexica, puzzle):
    target_bins = []
    for l in level:
        if l == "literal" or puzzle:
            target_bins.append(get_type_bin(type_idx, bins))
        else:
            target_bins.append(get_type_bin(type_idx + number_of_lexica, bins))
    return target_bins


def conv_frac(s):
    """
    converts fraction string to float

    """
    return float(sum(Fraction(x) for x in s.split()))