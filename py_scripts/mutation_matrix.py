import numpy as np
from tqdm import tqdm
from random import sample
from itertools import product
from py_scripts.player import LiteralPlayer,GriceanPlayer
from py_scripts.lexica import get_lexica,get_prior,get_lexica_bins

import sys 
import datetime
import csv
import os.path

"""Calculates the mutation_matrix"""

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

def summarize_counts(lst,states,messages):
    """ summarize counts for tuples of k-states and k-messages
    how many tuples were sampled of each pair

    :param lst: list containing state-message pairs
    :type lst: list
    :param states: amount of states
    :type states: int
    :param messages: amount of messages
    :type messages: int
    :return: list with state*messgae entries counter, with respect to lst
    :rtype: list
    """
    counter = [0 for _ in range(states*messages)]
    for i in range(len(lst)):
        s,m = lst[i][0] * messages, lst[i][1]
        counter[s+m] += 1
    return counter

def get_obs(all_states, all_messages, k,likelihoods,sample_amount):
    """Returns summarized counts of k-length <s_i,m_j> production observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], ...]] = k
    get production data from all types

    :type k: int
    :param likelihoods: list containig likelihood-lists (sender matrices)
    :type likelihoods: np.array
    :param sample_amount: number of required sequences
    :type sample_amount: int
    :return: list containing lists with numbers of sampled sm-pairs ([0,4,0,2], [3,4,3,2]) first sample 0 s_0,m_0, 4 s_1, m_1
    :rtype: list
    """
        
    obs = [] #store all produced k-length (s,m) sequences 

    for t in range(len(likelihoods)): # iterating over types
        produced_obs = [] #store k-length (s,m) sequences of a type
        type_m_amount = likelihoods[t].shape[1] # get size of type for doubled_state_freq
        type_s_amount = likelihoods[t].shape[0] 
        s = list(range(type_s_amount))
        m = list(range(type_m_amount))
        atomic_observations = list(product(s,m)) # all possible state message pairs
        production_vector = likelihoods[t].flatten()
        state_freq = np.ones(type_s_amount) / float(type_s_amount) #frequency of states s_1,...,s_n 
        doubled_state_freq = np.column_stack(tuple(state_freq for _ in range(type_m_amount))).flatten() # P(s)
        sample_vector = production_vector * doubled_state_freq # P(m|s,t_i) * P(s)

        for _ in range(sample_amount):
            sampled_idx = [np.random.choice(range(len(atomic_observations)),p=sample_vector) for _ in range(k)] #sample state_message pair
            sampled_obs = [atomic_observations[x] for x in sampled_idx]
            produced_obs.append(summarize_counts(sampled_obs, max(all_states), max(all_messages)))
        obs.append(produced_obs)
    return obs

def get_likelihood(all_states, all_messages, obs,likelihoods):
    """P(parent data|t_i) for all types and its samples

    :param obs: summarized counts of sampled ms-pairs for a type
    :type obs: np.array
    :param likelihoods: list containig sender matrices
    :type likelihoods: np.array
    :return: matrix with likelihood for all types
    :rtype: np.array
    """
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in range(len(likelihoods)):
        flat_lhi = likelihoods[lhi].flatten()

        for o in range(len(obs)):
            max_messages = max(all_messages)
            type_messages = likelihoods[lhi].shape[1]
            if type_messages < max_messages:
                m_counter = 1
                prod = []
                index = 0
                for x in range(len(obs[o])):
                    if m_counter <= type_messages:
                        prod.append(flat_lhi[index]**obs[o][x])
                        index += 1
                        m_counter += 1
                    elif m_counter < max_messages:
                        m_counter += 1
                    elif m_counter == max(all_messages):
                        m_counter = 1
                out[lhi,o] =  np.prod(prod) 
            else: 
                out[lhi,o] = np.prod([flat_lhi[x_i]**obs[o][x_i] for x_i in range(len(obs[o]))])
    return out

def get_mutation_matrix(s_amount,m_amount, likelihoods,lexica_prior,learning_parameter,sample_amount,k,lam,alpha,mutual_exclusivity, result_path, predefined):
    """Computes mutation matrix

    """
    if os.path.isfile('experiments/%s/matrices/qmatrix-s%s-m%s-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' 
                       %(result_path, str(s_amount),str(m_amount),lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity))) and not predefined:
        print('# Loading mutation matrix,\t', datetime.datetime.now().replace(microsecond=0))
        return np.genfromtxt('experiments/%s/matrices/qmatrix-s%s-m%s-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' 
               %(result_path, s_amount,str(m_amount),lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity)), delimiter=',')
    else:
        print('# Computing mutation matrix, ', datetime.datetime.now().replace(microsecond=0))
        obs = get_obs(s_amount, m_amount, k,likelihoods,sample_amount) # get production data from all types
        out = np.zeros([len(likelihoods),len(likelihoods)]) #matrix to store Q # len(likelihoods) = type amount

        for parent_type in tqdm(range(len(likelihoods))):
            type_obs = obs[parent_type] #Parent production data
            lhs = get_likelihood(s_amount, m_amount, type_obs,likelihoods) #P(parent data|t_i) for all types
            post = normalize(lexica_prior * np.transpose(lhs)) #P(t_j|parent data) for all types; P(t_j)*P(d|t_j)
            parametrized_post = normalize(post**learning_parameter)
            normed_lhs = lhs[parent_type] / np.sum(lhs[parent_type]) # norm P(parent_data|parent_type)
            out[parent_type] = np.dot(np.transpose(normed_lhs),parametrized_post)
    
        q = out
        with open('experiments/%s/matrices/qmatrix-s%s-m%s-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' %(result_path, str(s_amount),str(m_amount),lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity)),'w') as file:
            f_q = csv.writer(file)
            for i in q:
                f_q.writerow(i)
    
        return q