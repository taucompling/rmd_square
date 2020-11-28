import numpy as np
import math

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
    def __init__(self,lam,lexicon):
        """LiteralPlayer

        :param lam: lamda for softmax
        :type lam: int
        :param lexicon: lexicon
        :type lexicon: np.array
        """
        self.lam = lam
        self.lexicon = lexicon
        self.sender_matrix = self.sender_selection_matrix(lexicon)
        self.receiver_matrix =  self.receiver_selection_matrix()
        self.type = "LITERAL"

    def __repr__(self):
        return f"Type: {self.type}\nLex:\n{self.lexicon}\nSpeaker matrix:\n{self.sender_matrix}\nHearer matrix:\n{self.receiver_matrix}"

    def sender_selection_matrix(self,l):
        """speaker matrix: states in row, message in columns

        :param l: lexicon
        :type l: np.array
        :return: sender_matrix 
        :rtype: np.array
        """
        m = np.zeros(np.shape(l)) # empty lexicon
        for i in range(np.shape(l)[0]): # rows
            for j in range(np.shape(l)[1]): # columns
                m[i,j] = np.exp(self.lam * l[i,j]) 
        return normalize(m)

    def receiver_selection_matrix(self):
        """ hearer matrix: messages in rows, states in columns
        Takes transposed lexicon and normalize row-wise (prior over states plays no role as it's currently uniform)"""

        m = normalize(np.transpose(self.lexicon))

        for r in range(np.shape(m)[0]):

            if sum(m[r]) == 0 or math.isnan(sum(m[r])): # added isnan
                for c in range(np.shape(m)[1]):
                    m[r,c] = 1. / np.shape(m)[0] # split equally
        return m



class GriceanPlayer:
    def __init__(self,alpha, lam, lexicon):
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
        self.sender_matrix = self.sender_selection_matrix(lexicon)
        self.receiver_matrix = self.receiver_selection_matrix()
        self.type = "PRAGMATIC"

    def __repr__(self):
        return f"Type: {self.type}\nLex:\n{self.lexicon}\nSpeaker matrix:\n{self.sender_matrix}\nHearer matrix:\n{self.receiver_matrix}"
        
    def sender_selection_matrix(self,l):
        """pragmatic speaker, states in row, message in columns

        :param l: lexicon
        :type l: np.array
        :return: calculated matrix
        :rtype: np.array
        """
        literalListener = normalize(np.transpose(l))
        utils = np.transpose(literalListener)
        return normalize(np.exp(self.lam*np.power(utils, self.alpha)))

    def receiver_selection_matrix(self):
        """
        hearer matrix: messages in rows, states in columns
        Takes transposed sender matrix and normalize row-wise (prior over states plays no role as it's currently uniform)"""
        literalsender = np.zeros(np.shape(self.lexicon))
        for i in range(np.shape(self.lexicon)[0]):
            for j in range(np.shape(self.lexicon)[1]):
                literalsender[i,j] = np.exp(self.lam * self.lexicon[i,j])
        literalsender = normalize(literalsender)
        return normalize(np.transpose(literalsender))


