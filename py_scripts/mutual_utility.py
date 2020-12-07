
import numpy as np
from tqdm import tqdm
#np.set_printoptions(threshold=np.nan)
from random import sample
from itertools import product
from py_scripts.player import LiteralPlayer,GriceanPlayer
from py_scripts.lexica import get_lexica,get_prior,get_lexica_bins
from py_scripts.mutation_matrix import summarize_counts, get_obs, get_mutation_matrix, get_likelihood
import sys 
import datetime
import csv
import os.path

"""Calculates mutual utilities"""

def get_utils(typeList, all_messages, all_states, lam,alpha,mutual_exclusivity, result_path, predefined, state_priors):
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
    states = max(all_states)             
    messages = max(all_messages)
    state_priors = state_priors[..., None] 
    if os.path.isfile('experiments/%s/matrices/umatrix-s%s-m%s-lam%d-a%d-me%s.csv' %(result_path, str(all_states),str(all_messages),lam,alpha,str(mutual_exclusivity))) and not predefined:
        print('# Loading utilities,\t\t', datetime.datetime.now().replace(microsecond=0))
        return np.genfromtxt('experiments/%s/matrices/umatrix-s%s-m%s-lam%d-a%d-me%s.csv' %(result_path, str(all_states),str(all_messages),lam,alpha,str(mutual_exclusivity)),delimiter=',')
    else:
        print('# Computing utilities, ', datetime.datetime.now().replace(microsecond=0))
        out = np.zeros([len(typeList), len(typeList)])
        for i in tqdm(range(len(typeList))):
            for j in range(i, len(typeList)): # symmetric matrix, just iterated over half of it

                sender_i = typeList[i].sender_matrix
                sender_j = typeList[j].sender_matrix

                receiver_i = typeList[i].receiver_matrix
                receiver_j = typeList[j].receiver_matrix

                out[i,j] = np.sum(sender_i * np.transpose(receiver_j) * state_priors) + np.sum(sender_j * np.transpose(receiver_i)* state_priors)
                
                out[j,i] = out[i, j] 
 

    if not os.path.isdir("experiments/" + result_path +"/" + "matrices/"):
        os.makedirs("experiments/" + result_path + "/" + "matrices/" )
    with open("experiments/%s/matrices/umatrix-s%s-m%s-lam%d-a%d-me%s.csv" %(result_path, str(states), str(messages),lam,alpha,str(mutual_exclusivity)), "w") as file:

        f_u = csv.writer(file)
        for i in out:
            f_u.writerow(i)
    return out