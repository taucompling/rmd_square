import matplotlib.pyplot as plt
import numpy as np
from py_scripts.plots.x_best_prag_lit import find_best_x_prag_lit

def pragmatic_vs_literal(probs, results, lexica, result_path, x):
    plt.clf()
    x_best_literal, x_best_pragmatic = find_best_x_prag_lit(results, lexica, x)

    epochs_best_6_pragmatic = [[] for _ in range(len(x_best_pragmatic))]
    for run in probs:
        for t, prob in enumerate(run):
            if t in x_best_pragmatic:
                epochs_best_6_pragmatic[list(x_best_pragmatic).index(t)].append(prob)

    epochs_best_6_literal = [[] for _ in range(len(x_best_literal))]
    for run in probs:
        for t, prob in enumerate(run):
            if t in x_best_literal:
                epochs_best_6_literal[list(x_best_literal).index(t)].append(prob)

    plots1=list()
    plots2=list()
    for type, probs in enumerate(epochs_best_6_literal):
        plot,=plt.plot(probs, label=f"Type: {type}")
        plots1.append(plot)

    for type, probs in enumerate(epochs_best_6_pragmatic):
        plot,=plt.plot(probs, label=f"Type: {type}", linestyle="dashed")
        plots2.append(plot)
        
    plt.ylabel("proportion")
    plt.xlabel("epochs")

    legend1 = plt.legend(plots1, x_best_literal, title="literal", fontsize='small', fancybox=True, loc=1)
    plt.legend(plots2, x_best_pragmatic,title="pragmatic", fontsize='small', fancybox=True, loc=4)
    plt.gca().add_artist(legend1)
    plt.savefig(f'experiments/{result_path}/results/progress_best_{x}_lit_and_prag_types.png')