import numpy as np
from random import sample
from tqdm import tqdm
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

def get_obs(states, all_messages, k, typeList, sample_amount, negation_rate):
    """Returns summarized counts of k-length <s_i,m_j> production observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], ...]] = k
    get production data from all types

    :type k: int
    :param typeList: list containig all types 
    :type sender_matrices: list 
    :param sample_amount: number of required sequences
    :type sample_amount: int
    :return: list containing lists with numbers of sampled sm-pairs ([0,4,0,2], [3,4,3,2]) first sample 0 s_0,m_0, 4 s_1, m_1
    :rtype: list
    """
        
    obs = [] #store all produced k-length (s,m) sequences 
    sender_matrices = [t.sender_matrix for t in typeList]

    for t in range(len(sender_matrices)): # iterating over types
        produced_obs = [] #store k-length (s,m) sequences of a type
        type_m_amount = sender_matrices[t].shape[1] # get size of type for doubled_state_freq
        type_s_amount = sender_matrices[t].shape[0] 
        s = list(range(type_s_amount))
        m = list(range(type_m_amount))
        atomic_observations = list(product(s,m)) # all possible state message pairs
        production_vector = sender_matrices[t].flatten()
        state_freq = np.ones(type_s_amount) / float(type_s_amount) #frequency of states s_1,...,s_n 
        doubled_state_freq = np.column_stack(tuple(state_freq for _ in range(type_m_amount))).flatten() # P(s)
        sample_vector = production_vector * doubled_state_freq # P(m|s,t_i) * P(s)
        # print(sample_vector)

        for _ in range(sample_amount):
            sampled_idx = [np.random.choice(range(len(atomic_observations)),p=sample_vector) for _ in range(k)] #sample state_message pair
            sampled_obs = [atomic_observations[x] for x in sampled_idx]
            
            if not negation_rate:
                produced_obs.append(summarize_counts(sampled_obs, states, max(all_messages)))
            else:
                negated_sender_matrix = np.flip(sender_matrices[t], 0).flatten()
                negated_sample_vector = negated_sender_matrix * doubled_state_freq
                negated_k = int(k*negation_rate)
                negated_idx = [np.random.choice(range(len(atomic_observations)),p=negated_sample_vector) for _ in range(negated_k)] 
                normal_idx = [np.random.choice(range(len(atomic_observations)),p=sample_vector) for _ in range(k - negated_k)]
                mixed_obs = [atomic_observations[x] for x in normal_idx + negated_idx]
                produced_obs.append(summarize_counts(mixed_obs, states, max(all_messages)))
        obs.append(produced_obs)
    return obs

def get_likelihood(obs,sender_matrices, negation_rate):
    """P(parent data|t_i) for all types and its samples

    :param obs: summarized counts of sampled ms-pairs for a type
    :type obs: np.array
    :param sender_matrices: list containig sender matrices
    :type sender_matrices: np.array
    :return: matrix with likelihood for all types
    :rtype: np.array
    """
    out = np.zeros([len(sender_matrices), len(obs)]) # matrix to store results in
    for lhi in range(len(sender_matrices)):
        flat_sender = sender_matrices[lhi].flatten()
        #print(flat_lhi)

        for o in range(len(obs)):

            out[lhi,o] = np.prod([flat_sender[x_i]**obs[o][x_i] for x_i in range(len(obs[o]))])

                
    return out

def get_mutation_matrix(s_amount,m_amount, typeList,lexica_prior,learning_parameter,sample_amount,k,lam,alpha,mutual_exclusivity, result_path, state_priors, negation_rate):
    """Computes mutation matrix

    """

    if os.path.isfile('%s/matrices/qmatrix-%s-s%s-m%s-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' 
                       %(result_path, str(state_priors), str(s_amount),str(m_amount),lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity))):
        print('# Loading mutation matrix,\t', datetime.datetime.now().replace(microsecond=0))
        return np.genfromtxt('%s/matrices/qmatrix-%s-s%s-m%s-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' 
               %(result_path, str(state_priors), str(s_amount),str(m_amount),lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity)), delimiter=',')
    else:
        
        print('# Computing mutation matrix,\t', datetime.datetime.now().replace(microsecond=0))
        
        sender_matrices = [t.sender_matrix for t in typeList]

        obs = get_obs(s_amount, m_amount, k, typeList ,sample_amount, negation_rate) # get production data from all types
        out = np.zeros([len(sender_matrices),len(sender_matrices)]) #matrix to store Q # len(sender_matrices) = type amount
        
        for parent_type in range(len(sender_matrices)):

            type_obs = obs[parent_type] #Parent production data            

            lhs = get_likelihood(type_obs,sender_matrices, negation_rate) #P(parent data|t_i) for all types

            

            post = normalize(lexica_prior * np.transpose(lhs)) #P(t_j|parent data) for all types; P(t_j)*P(d|t_j)


            
            
            parametrized_post = normalize(post**learning_parameter)

            
            normed_lhs = lhs[parent_type] / np.sum(lhs[parent_type]) # norm P(parent_data|parent_type)

            out[parent_type] = np.dot(np.transpose(normed_lhs),parametrized_post)

        q = out
        with open('%s/matrices/qmatrix-%s-s%s-m%s-lam%d-a%d-k%d-samples%d-l%d-me%s.csv' %(result_path, str(state_priors), str(s_amount),str(m_amount),lam,alpha,k,sample_amount,learning_parameter,str(mutual_exclusivity)),'w') as file:
            f_q = csv.writer(file)
            for i in q:
                f_q.writerow(i)
    
        return q