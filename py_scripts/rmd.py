## Functions for replicator-mutator dynamics with iterated learning as mutator dynamics

import numpy as np
from tqdm import tqdm
#np.set_printoptions(threshold=np.nan)
from random import sample
from itertools import product
from py_scripts.player import LiteralPlayer,GriceanPlayer
from py_scripts.lexica import get_lexica,get_prior,get_lexica_bins
from py_scripts.process_predefined_lexica import  get_predefined_lexica
from py_scripts.mutation_matrix import summarize_counts, get_obs, get_mutation_matrix, get_likelihood
from py_scripts.mutual_utility import get_utils
from py_scripts.plots.plot_progress import plot_progress
from py_scripts.plots.pragmatic_vs_literal import pragmatic_vs_literal
from py_scripts.plots.x_best_prag_lit import print_best_x_types_to_file
from py_scripts.plots.proportion_target_types import get_target_types
import sys 
import datetime
import csv
import os.path

def get_type_bin(type_idx, bins):
    for b in range(len(bins)):
        if type_idx in bins[b]:
            return b

def get_lexica_representations(type, lexica):
    l = lexica[type-len(lexica)]
    for column in l.T:
        if -5 in column:
            l = np.delete(l, np.where(l == -5)[1][0], 1)
    return f"{check_literal_or_pragmatic(type, lexica)}:\n{l}"

def check_literal_or_pragmatic(type, lexica):
    if type > len(lexica)-1:
        return "pragmatic"
    else:
        return "literal"

def get_target_bins(type_idx, bins, level, number_of_lexica):
    target_bins = []
    for l in level:
        if l == "literal":
            target_bins.append(get_type_bin(type_idx, bins))
        else:
            target_bins.append(get_type_bin(type_idx + number_of_lexica, bins))
    return target_bins


def run_dynamics(alpha,lam,k,sample_amount,gens,runs,learning_parameter,kind,mutual_exclusivity, result_path, 
                cost, target_lex, target_level, competitor_lex, competitor_level, predefined = False, states=0, messages=0, print_x=6):

    """[Runs the replicator mutator dynamics
    :param alpha: rate to control difference between semantic and pragmatic violations
    :type alpha: int
    :param lam: softmax parameter
    :type lam: int
    :param k: sequence length
    :type k: int
    :param sample_amount: amount of samples (how many sequences)
    :type sample_amount: int
    :param gens: number of generations per simulation run
    :type gens: int
    :param runs: number of independent simulation runs
    :type runs: int
    :param learning_parameter: prob-matching = 1, increments approach MAP
    :type learning_parameter: int
    :param kind: model type (r (replicator), m (mutator), rmd)
    :type kind: string
    :param mutual_exclusivity: no concept assigned to more than one message
    :type mutual_exclusivity: boolean
    :param result_path: path to store results
    :type string
    :cost: type of cost function for the messages 
    :type: string
    :param target_lex: lexicon of target type
    :type: list
    :param target_level: level of target type (literal or pragmatic)
    :type: list
    :param competitor_lex: lexicon of competitor type
    :type: list
    :param competitor_level: level of competitor type (literal or pragmatic)
    :type: list
    :param predefined: Path to predefined lexica, else False
    :type: string or boolean
    :param states: amount of states
    :type states: int
    :param messages: amount of messages
    :type messages: int
    :param print_x: number of best x types to print in the results
    :type: int
    """
 
    print('# Starting,\t\t\t', datetime.datetime.now().replace(microsecond=0))
    if not os.path.isdir("experiments/" + result_path):
        os.makedirs("experiments/" + result_path)

    all_messages = messages
    all_states = states
    if predefined: 
        lexica, target_index, competitor_index, all_states, all_messages = get_predefined_lexica(predefined, target_lex, competitor_lex)


    else: 
        if len(target_lex[0]) not in all_messages or len(target_lex) not in all_states:
            raise Exception("Target type does not fit the number of states and messages!")
        if len(competitor_lex[0]) not in all_messages or len(competitor_lex) not in all_states:
            raise Exception("Competitor type does not fit the number of states and messages!")

        lexica, target_index, competitor_index = [], [], []
        for message in all_messages:
            for state in all_states:
                lexicon, target_in, competitor_in = get_lexica(state, message, max(all_messages), target_lex, competitor_lex, mutual_exclusivity)
                lexica += lexicon
                target_index += target_in
                competitor_index += competitor_in

    #for l in lexica:
    #    print(l)
    bins = get_lexica_bins(lexica, all_states) #To bin types with identical lexica
    target_bins = get_target_bins(target_index[0], bins, target_level, len(lexica))
    competitor_bins = get_target_bins(competitor_index[0], bins, competitor_level, len(lexica))

    l_prior = get_prior(lexica, cost, all_states)
    typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(alpha,lam,lex) for lex in lexica]
    likelihoods = [t.sender_matrix for t in typeList]

    
    u = get_utils(typeList, all_messages, all_states, lam,alpha,mutual_exclusivity, result_path, predefined)

    q = get_mutation_matrix(all_states, all_messages,likelihoods,l_prior,learning_parameter,sample_amount,k,lam,alpha,mutual_exclusivity, result_path, predefined)
    # raise Exception    
    print('# Beginning multiple runs,\t', datetime.datetime.now().replace(microsecond=0))

    if not os.path.isdir("experiments/" + result_path + "/results/"):
        os.makedirs("experiments/" + result_path + "/" +"results/")
    f = csv.writer(open('experiments/%s/results/%s-s%s-m%s-lam%d-a%d-k%d-samples%d-l%d-g%d-me%s.csv' %(result_path, kind,str(all_states),str(all_messages),lam,alpha,k,sample_amount,learning_parameter,gens,str(mutual_exclusivity)),'w'))
    f.writerow(['runID','kind']+['t_ini'+str(x) for x in range(len(typeList))] +\
               ['lam', 'alpha','k','samples','l','gens', 'm_excl'] + ['t_final'+str(x) for x in range(len(typeList))])
    

    if os.path.isfile('experiments/%s/results/00mean-%s-s%s-m%s-g%d-r%d-me%s.csv' %(result_path, kind,str(all_states),str(all_messages),gens,runs,str(mutual_exclusivity))):
        f_mean = csv.writer(open('experiments/%s/results/00mean-%s-s%s-m%s-g%d-r%d-me%s.csv' %(result_path, kind,str(all_states),str(all_messages),gens,runs,str(mutual_exclusivity)), 'a'))
    else: 
        f_mean = csv.writer(open('experiments/%s/results/00mean-%s-s%s-m%s-g%d-r%d-me%s.csv' %(result_path, kind,str(all_states),str(all_messages),gens,runs,str(mutual_exclusivity)), 'w'))
        f_mean.writerow(['kind','lam','alpha','k','samples','l','gens','runs','m_excl'] + ['t_mean'+str(x) for x in range(len(typeList))])
       

    p_sum = np.zeros(len(typeList)) #vector to store mean across runs
    
    winners = []
    progress = []
    for i in range(runs):
        p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
        p_initial = p
        for _ in range(gens):
            if kind == 'rmd':
                pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))] # type_prob * fitness
                pPrime = pPrime / np.sum(pPrime) # / average fitness in population 
                p = np.dot(pPrime, q) # * Q-matrix
            elif kind == 'm': # learnability
                p = np.dot(p,q) # without fitness
            elif kind == 'r': # communicative success
                pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))] # without mutation matrix
                p = pPrime / np.sum(pPrime)
    
        f.writerow([str(i),kind] + [str(p_initial[x]) for x in range(len(typeList))]+\
                   [str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens),str(mutual_exclusivity)] +\
                   [str(p[x]) for x in range(len(typeList))])
        p_sum += p
        p_i_mean = p_sum/(i+1)
        winners.append(np.argmax(p_i_mean))
        progress.append(p_i_mean)
    p_mean = p_sum / runs
    f_mean.writerow([kind,str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens),str(runs),str(mutual_exclusivity)] +\
                        [str(p_mean[x]) for x in range(len(typeList))])
    
    
    inc = np.argmax(p_mean)
    best_x = np.argpartition(p_mean, -print_x)[-print_x:]
    sorted_best_x = np.flip(best_x[np.argsort(p_mean[best_x])])
    inc_bin = get_type_bin(inc,bins)

    plot_progress(progress, sorted_best_x, result_path)
    pragmatic_vs_literal(progress, p_mean, lexica, result_path, print_x)
    print_best_x_types_to_file(p_mean, lexica,result_path, print_x)


    get_target_types(inc, bins, target_bins, competitor_bins, p_mean, result_path)

    end_results = [
    "-------------------------------------------------------------------------------------------------------------------------------------------------\n", 
    '*** Results with parameters: dynamics= %s, alpha = %d, lambda = %d, k = %d, samples per type = %d, learning parameter = %.2f, generations = %d, runs = %d ***\n' % (kind, alpha, lam, k, sample_amount, learning_parameter, gens, runs),
    f"*** Lexica parameters: states={all_states}, messages={all_messages}, cost={cost}***\n",
    "-------------------------------------------------------------------------------------------------------------------------------------------------\n",     
    f"# Incumbent type: {inc} with proportion {p_mean[inc]}\n",
    f"{get_lexica_representations(inc, lexica)}\n",
    f"# The bin of the incumbent {inc_bin}\n",
    f"# Bin: {bins[inc_bin]}\n",
    "-------------------------------------------------------------------------------------------------------------------------------------------------\n",    
    f"# 6 best types: \n"]
    for typ in sorted_best_x:
        end_results.append(f"-Type {typ} with proportion {p_mean[typ]}\n")
        end_results.append(f"{(get_lexica_representations(typ, lexica))}\n")
    # print("# All bins:", bins)
    
    print("# Finished:\t\t\t", datetime.datetime.now().replace(microsecond=0))
    for line in end_results:
        print(line)
    with open(f"experiments/{result_path}/results/end_results.txt", "w") as end:
        end.writelines(end_results)