import matplotlib.pyplot as plt
import numpy as np

def get_target_types(best, bins,  target_bins, competitor_bins, all_progress, result_path):
    #f"# Average Proportion: {round(bin_orders[bin_winner][0]/bin_orders[bin_winner][1],4)}\n",

    #print("TARGET", target_bins)
    #print("COMPETITOR", competitor_bins)

    plt.clf()
    target_types = bins[target_bins[0]]
    competitor_types = bins[competitor_bins[0]]
    runs = len(all_progress)
    target_probs, competitor_probs =  [[] for _ in range(runs)], [[] for _ in range(runs)]
    print(target_types)

    for run_index, run in enumerate(all_progress):
        last_generation = run[-1]
        for t, prob in enumerate(last_generation):            
            if t in target_types:
                target_probs[run_index].append(prob)
            if t in competitor_types:
                competitor_probs[run_index].append(prob)


    average_prob_target = np.average([np.max(prob) for prob in target_probs])
    average_prob_comp = np.average([np.max(prob) for prob in competitor_probs])

    target_types = [str(target_types)]
    target_proportions = [average_prob_target]
    plt.bar(target_types,target_proportions, color="maroon", label="Target Type")

    comp_types = [str(competitor_types)]
    comp_proportions = [average_prob_comp]
    plt.bar(comp_types,comp_proportions, color="blue", label="Competitor Type")

    plt.axhline(y=best,linewidth=1, color='k', linestyle="dashed",  label="Majority type")
    plt.ylabel("proportion")
    plt.xlabel("types")
    plt.legend()

    plt.savefig(f'{result_path}/results/proportion_targets_and_competitors.png')
    