
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

def pad_lexicon(l, states, messages, pad_value=-100):
    """adds pads to the lexicon

    :param l: lexicon
    :type l: np.array
    :param states: number of desired states after padding
    :type states: int
    :param messages: number of desired messages after padding
    :type messages: int
    :param pad_value: value used for padding, default=-100
    :type pad_value: int
    :return: padded lexicon
    :rtype: np.array
    """
    return np.pad(l, [(0, states - l.shape[0]),(0,messages - l.shape[1])], mode='constant', constant_values=pad_value) 

def get_utils(typeList,states,messages,lam,alpha,mutual_exclusivity, result_path, predefined):
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

    if os.path.isfile('experiments/%s/matrices/umatrix-s%s-m%s-lam%d-a%d-me%s.csv' %(result_path, str(states),str(messages),lam,alpha,str(mutual_exclusivity))) and not predefined:
        print('# Loading utilities,\t\t', datetime.datetime.now().replace(microsecond=0))
        return np.genfromtxt('experiments/%s/matrices/umatrix-s%s-m%s-lam%d-a%d-me%s.csv' %(result_path, str(states),str(messages),lam,alpha,str(mutual_exclusivity)),delimiter=',')
    else:
        print('# Computing utilities, ', datetime.datetime.now().replace(microsecond=0))
        out = np.zeros([len(typeList), len(typeList)])
        for i in tqdm(range(len(typeList))):
            for j in range(len(typeList)):
                pad_sender_i = pad_lexicon(typeList[i].sender_matrix, states, messages, 1)
                pad_sender_j = pad_lexicon(typeList[j].sender_matrix, states, messages, 1)

                pad_receiver_i = np.transpose(pad_lexicon(np.transpose(typeList[i].receiver_matrix), states, messages, 1)) #TODO mit was padden?
                pad_receiver_j = np.transpose(pad_lexicon(np.transpose(typeList[j].receiver_matrix), states, messages, 1))

                #print("PAD SENDER I", pad_sender_i)
                #print("PAD SENDER J", pad_sender_j)
                out[i,j] = (np.sum(pad_sender_i * np.transpose(pad_receiver_j))  / states 
                            + np.sum(pad_sender_j * np.transpose(pad_receiver_i))/ states ) / 2 
            #raise Exception
    if not os.path.isdir("experiments/" + result_path +"/" + "matrices/"):
        os.makedirs("experiments/" + result_path + "/" + "matrices/" )
    with open("experiments/%s/matrices/umatrix-s%s-m%s-lam%d-a%d-me%s.csv" %(result_path, str(states), str(messages),lam,alpha,str(mutual_exclusivity)), "w") as file:

        f_u = csv.writer(file)
        for i in out:
            f_u.writerow(i)
    return out