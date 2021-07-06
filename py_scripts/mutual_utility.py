import numpy as np
from tqdm import tqdm
#np.set_printoptions(threshold=np.nan)
import datetime
import csv
import os.path

"""Calculates mutual utilities"""

def get_utils(typeList, all_messages, states, lam,alpha,mutual_exclusivity, result_path, state_priors, utility_message_cost):
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
    state_priors = np.array(state_priors)
    state_priors = state_priors[..., None] # change row to column
    
    if os.path.isfile("%s/matrices/umatrix-%s-s%s-m%s-lam%d-a%d-me%s-umc%s.csv" %(result_path, str(state_priors.flatten()), str(states), str(messages),lam,alpha,str(mutual_exclusivity), utility_message_cost)):
        print('# Loading utilities,\t\t', datetime.datetime.now().replace(microsecond=0))
        return np.genfromtxt("%s/matrices/umatrix-%s-s%s-m%s-lam%d-a%d-me%s-umc%s.csv" %(result_path, str(state_priors.flatten()), str(states), str(messages),lam,alpha,str(mutual_exclusivity), utility_message_cost),delimiter=',')
        
    else:
        print('# Computing utilities,\t\t', datetime.datetime.now().replace(microsecond=0))
        out = np.zeros([len(typeList), len(typeList)])
        for i in range(len(typeList)):
            #print(f"# {round((i/len(typeList)) * 100)}% - " ,i, f"/{len(typeList)} utilities", '', end='\r')
            for j in range(i, len(typeList)): 

                sender_i = typeList[i].sender_matrix if not utility_message_cost else typeList[i].costly_sender_matrix
                sender_j = typeList[j].sender_matrix if not utility_message_cost else typeList[j].costly_sender_matrix

                receiver_i = typeList[i].receiver_matrix if not utility_message_cost else typeList[i].costly_receiver_matrix
                receiver_j = typeList[j].receiver_matrix if not utility_message_cost else typeList[j].costly_receiver_matrix

                out[i,j] = np.sum(sender_i * np.transpose(receiver_j) * state_priors) + np.sum(sender_j * np.transpose(receiver_i)* state_priors)
                out[j,i] = out[i, j] 


    if not os.path.isdir(result_path +"/" + "matrices/"):
        os.makedirs(result_path + "/" + "matrices/" )


    with open("%s/matrices/umatrix-%s-s%s-m%s-lam%d-a%d-me%s-umc%s.csv" %(result_path, str(state_priors.flatten()), str(states), str(messages),lam,alpha,str(mutual_exclusivity), utility_message_cost), "w") as file:

        f_u = csv.writer(file)
        for i in out:
            f_u.writerow(i)
    return out