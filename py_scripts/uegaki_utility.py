from os import EX_CANTCREAT
from copy import deepcopy
import numpy as np
from tqdm import tqdm
#np.set_printoptions(threshold=np.nan)
import datetime
import csv
import os.path

"""Calculates mutual utilities"""

def get_utils(typeList, all_messages, states, lam,alpha,mutual_exclusivity, result_path, predefined, state_priors, utility_message_cost):
    """calculates expected utiliy of types

    :param typeList: list containing already instantiated types of players
    :type typeList: list
    :param states: amount of states
    :type states: int
    :param messages: amount of messages
    :type messages: int
    :param lam: lambda
    :type lam: int
    :param alpha: softmax param
    :type alpha: int
    :param mutual_exclusivity: if mutual exclusive
    :type mutual_exclusivity: boolean
    :return: utility matrix
    :rtype: np.array
    """
             
    messages = max(all_messages)
    # state_priors = state_priors[..., None] # change row to column
    
    if os.path.isfile("experiments/%s/matrices/umatrix-%s-s%s-m%s-lam%d-a%d-me%s-umc%s.csv" %(result_path, str(state_priors.flatten()), str(states), str(messages),lam,alpha,str(mutual_exclusivity), utility_message_cost)) and not predefined:
        print('# Loading utilities,\t\t', datetime.datetime.now().replace(microsecond=0))
        return np.genfromtxt("experiments/%s/matrices/umatrix-%s-s%s-m%s-lam%d-a%d-me%s.csv" %(result_path, str(state_priors.flatten()), str(states), str(messages),lam,alpha,str(mutual_exclusivity)),delimiter=',')
    else:
        print('# Computing utilities,\t\t', datetime.datetime.now().replace(microsecond=0))

        out = np.zeros(len(typeList))
        for i in range(len(typeList)):

            lex = deepcopy(typeList[i].lexicon)
            lex = np.array([row for row in lex.T if -5 not in row]).T
            
            
            informativeness = 0
            for world_index, prior in enumerate(state_priors):
                sum_p_cw = 0
                world = lex[world_index]

                for message_index, message in enumerate(lex.T):
                    p_wc = lex[world_index, message_index] / np.sum(message) if np.sum(message) != 0 else 0
                    sum_p_wc = p_wc
                    p_cw = lex[world_index, message_index] / np.sum(world) if np.sum(world) != 0 else 0
                    sum_p_cw += p_cw * sum_p_wc
                sum_p_w =  prior * sum_p_cw                    
                informativeness += sum_p_w      
            out[i] = informativeness



    if not os.path.isdir("experiments/" + result_path +"/" + "matrices/"):
        os.makedirs("experiments/" + result_path + "/" + "matrices/" )
    with open("experiments/%s/matrices/umatrix-%s-s%s-m%s-lam%d-a%d-me%s-umc%s.csv" %(result_path, str(state_priors.flatten()), str(states), str(messages),lam,alpha,str(mutual_exclusivity), utility_message_cost), "w") as file:

        f_u = csv.writer(file)
        for i in out:
            f_u.writerow([i])
    return out