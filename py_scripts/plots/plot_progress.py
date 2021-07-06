import matplotlib.pyplot as plt
import numpy as np

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

def plot_progress(probs, print_x,result_path, lexica, puzzle):

    best_x = np.argpartition(probs[0], -print_x)[-print_x:]

    words = {(0, 1, 1): "some", (1, 0, 0): "none", (0, 0, 1): "all", (1, 1, 0): "not all"}


    best_x_lexica = []
    for x in best_x:
        lex = []
        l = lexica[x-len(lexica)]
        for column in l.T:
            if -5 in column:
                l = np.delete(l, np.where(l == -5)[1][0], 1)
            else:
                lex.append(words[tuple(column)])
        best_x_lexica.append(", ".join(lex))

    epochs_best_x = [[] for _ in range(len(best_x))]

    for run in probs:
        for t, prob in enumerate(run):
            if t in best_x:
                epochs_best_x[list(best_x).index(t)].append(prob)

    for type, probs in enumerate(epochs_best_x):
        plt.plot(probs, label=f"Type: {type}")
    
    plt.ylabel("proportion")
    #plt.yscale('log')
    plt.xlabel("generations")
    #plt.legend(best_x)
    plt.legend(best_x_lexica)
    plt.savefig(f'{result_path}/results/progress_best_all_{len(best_x)}_types.png')
    
    