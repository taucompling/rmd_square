# import libraries
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import datetime
import csv
import os.path
import sys

# import other scripts
from py_scripts.player import LiteralPlayer,GriceanPlayer
from py_scripts.lexica import get_lexica,get_prior,get_lexica_bins
from py_scripts.mutation_matrix import get_mutation_matrix
from py_scripts.mutual_utility import get_utils as brochhagen_utils
from py_scripts.mutual_utility import get_utils as uegaki_utils
from py_scripts.message_costs import calculate_cost_dict
from py_scripts.utils import print_end_results, get_type_bin, get_lexica_representations, get_target_bins
from py_scripts.checks import check_state_priors, check_print_x

# import plotting scripts
from py_scripts.plots.informativesness_score import get_informativeness
from py_scripts.plots.plot_progress import plot_progress
from py_scripts.plots.plot_all_sizes import plot_all_sizes
from py_scripts.plots.x_best_prag_lit import print_best_x_types_to_file
from py_scripts.plots.proportion_target_types import get_target_types


def run_dynamics(general_settings, states_and_messages, models, other_features, storing_results, plotting_info, grid_state_priors=False):

    """Runs the replicator mutator dynamics"""

    print('# Starting,\t\t\t', datetime.datetime.now().replace(microsecond=0))

    # setting parameters 
    alpha, lam, k  = general_settings["alpha"], general_settings["lam"], general_settings["k"]
    sample_amount, learning_parameter = general_settings["sample_amount"], general_settings["learning_parameter"]
    gens, runs = general_settings["gens"], general_settings["runs"]

    states, messages = states_and_messages["states"], states_and_messages["messages"]
    kind, cost, mutual_utility_calculation = models["kind"], models["cost"], models["mutual_utility_calculation"]

    state_priors = check_state_priors(other_features["state_priors"], states, grid_state_priors)
    utility_message_cost, negation_rate = other_features["utility_message_cost"], other_features["negation_rate"]
    puzzle, only_prag, mutual_exclusivity = other_features["puzzle"], other_features["only_prag"], other_features["mutual_exclusivity"]

    result_path = storing_results["result_path"]
    target_lex, target_level =  plotting_info["target_lex"], plotting_info["target_level"]
    competitor_lex, competitor_level = plotting_info["competitor_lex"], plotting_info["competitor_level"]   
    print_x = check_print_x(plotting_info["print_x"],target_lex, competitor_lex, messages, states)

    # creating the folders needed
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path + "/results/"):
        os.makedirs(result_path + "/" + "results/")
    if not os.path.isdir(result_path + "/res_data/"):
        os.makedirs(result_path + "/" + "res_data/")

    # get lexica
    lexica, target_index, competitor_index = [], [], []
    for message in messages:
        lexicon, target_in, competitor_in = get_lexica(states, message, max(messages), target_lex, competitor_lex, mutual_exclusivity, puzzle)
        lex_len = len(lexica)
        lexica += lexicon
        target_index += [lex_len + t for t in target_in]
        competitor_index += [lex_len + t for t in competitor_in]

    # get  bins
    bins = get_lexica_bins(lexica, states, puzzle) # bin types with identical lexica
    target_bins = get_target_bins(target_index[0], bins, target_level, len(lexica), puzzle) if print_x > 0 else None
    competitor_bins = get_target_bins(competitor_index[0], bins, competitor_level, len(lexica), puzzle) if print_x > 0 else None

    # get costs and prios
    message_costs = calculate_cost_dict(cost, states, puzzle)   
    l_prior = get_prior(lexica, puzzle, message_costs) 

    # get type_list
    if only_prag:
        typeList = [GriceanPlayer(alpha,lam,lex, state_priors, message_costs) for lex in lexica]
    else:
        typeList = [LiteralPlayer(alpha,lam,lex, state_priors, message_costs) for lex in lexica] + [GriceanPlayer(alpha,lam,lex, state_priors, message_costs) for lex in lexica]
        
    # calculation of mutual utility u    
    if mutual_utility_calculation == "brochhagen":
        u = brochhagen_utils(typeList, messages, states, lam,alpha,mutual_exclusivity, result_path,state_priors, utility_message_cost)
    elif mutual_utility_calculation == "uegaki":
        u = uegaki_utils(typeList, messages, states, lam,alpha,mutual_exclusivity, result_path, state_priors, utility_message_cost)
    else:
        raise NotImplementedError

    # calculation of mutation matrix if needed
    if kind == "rmd" or kind == "m":
        q = get_mutation_matrix(states, messages, typeList,l_prior,learning_parameter,sample_amount,k,lam,alpha,mutual_exclusivity, result_path, state_priors, negation_rate)
   
    # preparing to store results
    f = csv.writer(open(f'{result_path}/res_data/{kind}-{state_priors}-s{states}-m{messages}-lam{lam}-a{alpha}-k{k}-samples{sample_amount}-l{learning_parameter}-g{gens}-me{mutual_exclusivity}-sp{state_priors}-puzzle{puzzle}.csv','w'))

    f.writerow(['runID','kind']+['t_ini'+str(x) for x in range(len(typeList))] +\
               ['lam', 'alpha','k','samples','l','gens', 'm_excl'] + ['t_final'+str(x) for x in range(len(typeList))])
    

    if os.path.isfile('%s/res_data/00mean-%s-%s-s%s-m%s-g%d-r%d-me%s.csv' %(result_path, kind, str(state_priors),str(states),str(messages),gens,runs,str(mutual_exclusivity))):
        f_mean = csv.writer(open('%s/res_data/00mean-%s-%s-s%s-m%s-g%d-r%d-me%s.csv' %(result_path, kind, str(state_priors),str(states),str(messages),gens,runs,str(mutual_exclusivity)), 'w'))
    else: 
        f_mean = csv.writer(open('%s/res_data/00mean-%s-%s-s%s-m%s-g%d-r%d-me%s.csv' %(result_path, kind, str(state_priors), str(states),str(messages),gens,runs,str(mutual_exclusivity)), 'w'))
        f_mean.writerow(['kind','lam','alpha','k','samples','l','gens','runs','m_excl'] + ['t_mean'+str(x) for x in range(len(typeList))])
       
    
    # starts to calculate generations and runs    
    print('# Beginning multiple runs,\t', datetime.datetime.now().replace(microsecond=0))
    p_sum = np.zeros(len(typeList)) # vector to store mean across runs

    gen_winners = [[None], 0] # to count winners in generations
    avg_gens = []
    bin_orders = defaultdict(lambda: [0,0])
    all_progress = []
    
    for r in range(runs): # runs
        progress = []
        p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
        p_initial = p
        gen = 0
        while True: # generations (at least gens, max 20.000 )
            if kind == 'rmd':
                fitness = [np.sum(u[t,] * p)  for t in range(len(typeList))] 
                pPrime = p * fitness # type_prob * fitness
                pPrime = pPrime / np.sum(pPrime) # average fitness in population 
                p = np.dot(pPrime, q) # * Q-matrix

            elif kind == 'm': # learnability
                p = np.dot(p,q) # without fitness

            elif kind == 'r': # communicative success
                pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))] # without mutation matrix

                p = pPrime / np.sum(pPrime)

            progress.append(p)

            gen_winner = np.argpartition(p, -print_x)[-print_x:] # ascending order
            sorted_gen_winner = np.flip(gen_winner[np.argsort(p[gen_winner])]) # descending order
            sorted_gen_winner_tuples = [(winner, round(p[winner], 10)) for winner in sorted_gen_winner]
            
            if np.array_equal(sorted_gen_winner_tuples, gen_winners[0]):
                gen_winners[1] +=1
            else:
                gen_winners[0], gen_winners[1] = sorted_gen_winner_tuples, 1
            
            if gen >= gens and gen_winners[1] >= 10: # stops if 10 times the same winner with same proportion
                avg_gens.append(gen)
                break
            
            if gen == 20000:
                avg_gens.append(gen)
                print("STOPPED after 20000 generations")
                break

            gen+=1

        all_progress.append(progress) 

        f.writerow([str(r),kind] + [str(p_initial[x]) for x in range(len(typeList))]+\
                   [str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens),str(mutual_exclusivity)] +\
                   [str(p[x]) for x in range(len(typeList))])
        p_sum += p
        
        # Track Winner Bin
        bin_orders[tuple(bins[get_type_bin(np.argmax(p), bins)])][0] += max(p)
        bin_orders[tuple(bins[get_type_bin(np.argmax(p), bins)])][1] += 1


    
    # Evaluation 

    gens = np.average(avg_gens)
    p_mean = p_sum / runs 
    f_mean.writerow([kind,str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens),str(runs),str(mutual_exclusivity)] +\
                        [str(p_mean[x]) for x in range(len(typeList))])
    

    # getting best types
    inc = np.argmax(p_mean)
    inc_bin = get_type_bin(inc,bins)
    sum_winning_types = 0
    for inc_bin_type in bins[inc_bin]:
        sum_winning_types += p_mean[inc_bin_type]
    bin_winner = max(bin_orders, key=lambda k: bin_orders[k][1])

    if grid_state_priors:
        if not os.path.isdir("grid_state_priors_results"):
            os.makedirs("grid_state_priors_results")
        if tuple(bins[target_bins[0]]) in bin_orders:
            with open("grid_state_priors_results/grid_results.txt", "a+") as w:
                prop = bin_orders[tuple(bins[target_bins[0]])][0]
                r = bin_orders[tuple(bins[target_bins[0]])][1]
                w.write(f"STATE PRIORS {state_priors}\n")
                w.write(f"Is max? {tuple(bins[target_bins[0]]) == bin_winner}\n")
                w.write(f"Percentage runs {round(r/runs, 2)}\n")
                w.write(f"Proportion {round(prop/r, 2)}\n")
                w.write(f"________________________\n")


        os.system("rm -rf experiments")
        return

    # plotting things
    get_informativeness(pPrime, typeList, bins, result_path)

    if print_x > 0:
        plot_all_sizes(progress, print_x, result_path, lexica, puzzle)
        # plot_progress(progress, print_x, result_path, lexica, puzzle)                 
        print_best_x_types_to_file(p_mean, lexica,result_path, print_x, puzzle)
        average_best = round(bin_orders[bin_winner][0]/bin_orders[bin_winner][1],4)
        # get_target_types(average_best, bins, target_bins, competitor_bins, all_progress, result_path)


    best_x = np.argpartition(p_mean, -print_x)[-print_x:] if print_x > 0 else np.argpartition(p_mean, -10)[-10:] # ascending order
    sorted_best_x = np.flip(best_x[np.argsort(p_mean[best_x])]) #descending order


    end_results = print_end_results(bin_orders, bin_winner, runs, lexica, puzzle, sum_winning_types, print_x)

    for typ in sorted_best_x:
        end_results.append(f"-Type {typ} with proportion {p_mean[typ]}\n")
        end_results.append(f"{(get_lexica_representations(typ, lexica, puzzle))}\n")
    
    print("# Finished:\t\t\t", datetime.datetime.now().replace(microsecond=0))
    for line in end_results:
        print(line)
    with open(f"{result_path}/results/end_results.txt", "w") as end:
        end.writelines(end_results)

    #os.system("rm -rf result_path")