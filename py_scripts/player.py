import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
# from message_costs import calculate_cost_dict
from copy import deepcopy

def normalize(m):
    """normalizes over rows
    if 0/0, return nan

    :param m: matrix/lexicon
    :type m: np.array
    :return: normalized matrix
    :rtype: np.array
    """

    m = m / m.sum(axis=1)[:, np.newaxis]
    return m 

class LiteralPlayer:
    def __init__(self,lam,lexicon, state_priors):
        """LiteralPlayer

        :param lam: lamda for softmax
        :type lam: int
        :param lexicon: lexicon
        :type lexicon: np.array
        """
        self.lam = lam
        self.lexicon = lexicon
        self.state_priors = state_priors
        self.sender_matrix = self.costly_sender_matrix = self.sender_selection_matrix() # costs don't make a difference for Literal Speaker
        self.receiver_matrix = self.costly_receiver_matrix = self.receiver_selection_matrix()
        self.type = "LITERAL"
        

    def __repr__(self):
        return f"Type: {self.type}\nLex:\n{self.lexicon}\nSpeaker matrix:\n{self.sender_matrix}\nHearer matrix:\n{self.receiver_matrix}"

    def sender_selection_matrix(self):
        """speaker matrix: states in row, message in columns

        :param l: lexicon
        :type l: np.array
        :return: sender_matrix 
        :rtype: np.array
        """
        l = self.lexicon
        m = np.zeros(np.shape(l)) # empty lexicon
        for i in range(np.shape(l)[0]): # rows
            for j in range(np.shape(l)[1]): # columns
                if l[i,j] == -5:
                    m[i,j] = 0
                else:
                    m[i,j] = np.exp(self.lam * l[i,j]) 
        return normalize(m)

    def receiver_selection_matrix(self):
        """ hearer matrix: messages in rows, states in columns"""

        weighted_lex = np.zeros(np.shape(self.lexicon))
        for i, row in enumerate(self.lexicon):
            weighted_lex[i] = float(self.state_priors[i]) * row

        m = normalize(np.transpose(weighted_lex))
        for r in range(np.shape(m)[0]):
            if sum(m[r]) == 0 or math.isnan(sum(m[r])): # added isnan
                m[r] = self.state_priors


        return m



class GriceanPlayer:
    def __init__(self,alpha, lam, lexicon, state_priors, costs):
        """
        :param alpha: rate to control difference between semantic and pragmatic violations
        :type alpha: int
        :param lam: lambda
        :type lam: int
        :param lexicon: lexicon
        :type lexicon: np.arrays
        """
        self.alpha = alpha
        self.lam = lam
        self.lexicon = lexicon
        self.state_priors = state_priors
        self.sender_matrix = self.sender_selection_matrix()
        self.receiver_matrix = self.receiver_selection_matrix()
        self.costly_sender_matrix = self.get_costly_sender_matrix(costs)
        self.costly_receiver_matrix = self.get_costly_hearer_matrix(costs)
        self.type = "PRAGMATIC"

    def __repr__(self):
        return f"Type: {self.type}\nLex:\n{self.lexicon}\nSpeaker matrix:\n{self.sender_matrix}\nHearer matrix:\n{self.receiver_matrix}"
        
    def sender_selection_matrix(self):
        """pragmatic speaker, states in row, message in columns
        :return: calculated matrix
        :rtype: np.array
        """
        l = self.lexicon
        literalListener = normalize(np.transpose(l))
        utils = np.transpose(literalListener)
        sender_sel = np.exp(self.lam*np.power(utils, self.alpha))

        for row in range(sender_sel.shape[0]):
            for column in range(sender_sel.shape[1]):
                if l[row, column] == -5:
                    sender_sel[row, column] = 0 
        return normalize(sender_sel)

    def get_costly_sender_matrix(self, costs):
        sm = deepcopy(self.sender_matrix.transpose())
        for c, column in enumerate(self.lexicon.transpose()):
            if -5 in column:
                continue
            sm[c] *= costs[tuple(column)]
        return normalize(sm.transpose())

    def receiver_selection_matrix(self):
        """
        hearer matrix: messages in rows, states in columns"""

        #print(self.lexicon)
        literalsender = np.zeros(np.shape(self.lexicon))
        for i in range(np.shape(self.lexicon)[0]):
            for j in range(np.shape(self.lexicon)[1]): 
                if self.lexicon[i,j] == -5:
                    literalsender[i,j] = 0 
                else:               
                    literalsender[i,j] = np.exp(self.lam * self.lexicon[i,j])

        for i, row in enumerate(literalsender):
            literalsender[i] = float(self.state_priors[i]) * row

        rec_sel = normalize(np.transpose(literalsender))
        for row in range(rec_sel.shape[0]):
            if sum(rec_sel[row]) == 0 or math.isnan(sum(rec_sel[row])):
                rec_sel[row] = self.state_priors

        return rec_sel

    def get_costly_hearer_matrix(self, costs):
        hm = deepcopy(self.costly_sender_matrix)
        for i, row in enumerate(hm):
            hm[i] = float(self.state_priors[i]) * row
            
        costly_hm = normalize(np.transpose(hm))
        for row in range(costly_hm.shape[0]):
            if sum(costly_hm[row]) == 0 or math.isnan(sum(costly_hm[row])):
                costly_hm[row] = self.state_priors

        #print(costly_hm)
        return costly_hm



#costs = calculate_cost_dict("brochhagen", 3, True)
#lex = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
#x = GriceanPlayer(1, 1, lex, [1/3, 1/3, 1/3], costs)
#print(x.sender_matrix)
