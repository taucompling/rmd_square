import matplotlib.pyplot as plt
import numpy as np

def plot_progress(probs, print_x,result_path):

    best_x = np.argpartition(probs[0], -print_x)[-print_x:]

    epochs_best_x = [[] for _ in range(len(best_x))]

    for run in probs:
        for t, prob in enumerate(run):
            if t in best_x:
                epochs_best_x[list(best_x).index(t)].append(prob)

    for type, probs in enumerate(epochs_best_x):
        plt.plot(probs, label=f"Type: {type}")
    plt.ylabel("proportion")
    plt.xlabel("generations")
    plt.legend(best_x)
    plt.savefig(f'experiments/{result_path}/results/progress_best_{len(best_x)}_types.png')
    
    