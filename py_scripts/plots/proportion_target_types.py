import matplotlib.pyplot as plt
def get_target_types(best, bins,  target_bins, competitor_bins, p_mean, result_path):
    plt.clf()
    target_types = ["T" + str(i+1) for i in range(len(["T" for bin in target_bins for typ in bins[bin]]))]
    target_proportions = [p_mean[typ] for bin in target_bins for typ in bins[bin]]
    plt.bar(target_types,target_proportions, color="maroon", label="Target Type")

    comp_types = ["C" + str(i+1) for i in range(len(["C" for bin in competitor_bins for typ in bins[bin]]))]
    comp_proportions = [p_mean[typ] for bin in competitor_bins for typ in bins[bin]]
    plt.bar(comp_types,comp_proportions, color="blue", label="Competitor Type")

    plt.axhline(y=p_mean[best],linewidth=1, color='k', linestyle="dashed",  label="Majority type")
    plt.ylabel("proportion")
    plt.xlabel("types")
    plt.legend()

    plt.show()
    plt.savefig(f'experiments/{result_path}/results/proportion_targets_and_competitors.png')