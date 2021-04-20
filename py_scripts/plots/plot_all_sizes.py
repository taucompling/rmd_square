import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from itertools import cycle
from cycler import cycler

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

def plot_all_sizes(probs, print_x,result_path, lexica, puzzle):

    sorted_types = defaultdict(list)

    for t, x in enumerate(probs[-1]):
        lex = []
        l = lexica[t-len(lexica)]
        for column in l.T:
            if -5 in column:
                l = np.delete(l, np.where(l == -5)[1][0], 1)

        sorted_types[l.shape[1]].append((t,x)) # sorted by size of lexicon

    amount = print_x // len(sorted_types) if print_x // len(sorted_types) > 0 else 1

    best_x = []
    for size, info in sorted_types.items():
        best_from_typ = sorted(info, key=lambda x: x[1])[-amount:]
        best_x.append(best_from_typ)

    words = {(0, 1, 1): "some", (1, 0, 0): "none", (0, 0, 1): "all", (1, 1, 0): "not all", 
              (1, 0, 0, 0): "and", (1, 1, 1, 0): "or", (0, 1, 1, 1): "nand",  (0, 0, 0, 1): "nor"}


    best_x_lexica = []
    for size in best_x:
        for typ in size:
            x, prob  = typ[0], typ[1]
            lex = []
            l = lexica[x-len(lexica)]
            for column in l.T:
                if -5 in column:
                    l = np.delete(l, np.where(l == -5)[1][0], 1)
                else:
                    lex.append(words[tuple(column)])
            best_x_lexica.append(", ".join(lex))

    best_x = [typ[0] for size in best_x for typ in size]

    epochs_best_x = [[] for _ in range(len(best_x))]

    for run in probs:
        for t, prob in enumerate(run):
            if t in best_x:
                epochs_best_x[list(best_x).index(t)].append(prob)

    lines = ["k-","k--","k-.","k:"]
    linecycler = cycle(lines)


    for type, probs in enumerate(epochs_best_x):
        plt.plot(probs, next(linecycler), label=f"Type: {type}")

    plt.ylabel("proportion")
    #plt.yscale('log')
    plt.xlabel("generations")
    plt.legend(best_x_lexica)
    plt.savefig(f'experiments/{result_path}/results/progress_best_{len(best_x)}_types.png')
    
    