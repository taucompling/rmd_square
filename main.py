from py_scripts.rmd import run_dynamics
from itertools import product
import numpy as np
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('yaml_path', metavar='path', type=str, help='path to yaml file')
args = parser.parse_args()

with open(args.yaml_path) as y:
    conf = yaml.load(y.read(), Loader=yaml.FullLoader)

state_priors = conf["RMD"]["other_features"]["state_priors"]

if state_priors != "grid":
    run_dynamics(**conf["RMD"])

else:
    states = conf["RMD"]["states_and_messages"]["states"]

    t1 = [round(0 + (x * 0.01),2) for x in range(0, 101)] 
    pro = list(product(t1, repeat=states))

    state_priors_list = [p for p in pro if np.sum(p) == 1 ]

    for sp in state_priors_list:
        run_dynamics(**conf["RMD"], grid_state_priors=sp)






