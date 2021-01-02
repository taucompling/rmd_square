from py_scripts.rmd_grid import run_dynamics
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('yaml_path', metavar='path', type=str, help='path to yaml file')
args = parser.parse_args()


t1 = [round(0 + (x * 0.01),2) for x in range(0, 101)] 
t2 = t1
t3 = t1

state_priors_list = []

for n1 in t1:
    for n2 in t2:
        for n3 in t3:
            if n1 + n2 + n3 == 1:
                state_priors_list.append([n1, n2, n3])


for sp in state_priors_list:
    with open(args.yaml_path) as y:
        conf = yaml.load(y.read(), Loader=yaml.FullLoader)
    run_dynamics(**conf["General setting"], state_priors=sp)







