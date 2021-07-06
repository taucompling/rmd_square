import numpy as np
from py_scripts.utils import conv_frac

def check_state_priors(state_priors, states, grid_state_priors):
    if grid_state_priors:
        state_priors = grid_state_priors
    elif not state_priors:
        state_priors = np.ones(states) / float(states)
    else:
        state_priors = np.array([conv_frac(sp) if type(sp) == str else sp for sp in state_priors  ])
        if state_priors.shape[0] != states:
            raise Exception(f"State priors don't fit the amount of states ({states})")
        if round(np.sum(state_priors), 6) != 1:
            raise Exception(f"State priors dont' sum up to 1.")
    return state_priors

def check_print_x(print_x,target_lex, competitor_lex, messages, states):
    if print_x > 0:
        if len(target_lex[0]) not in messages or len(target_lex) != states:
            raise Exception("Target type does not fit the number of states and messages!")
        if len(competitor_lex[0]) not in messages or len(competitor_lex)  != states:
            raise Exception("Competitor type does not fit the number of states and messages!")
        return print_x
    return print_x