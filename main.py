from py_scripts.rmd import run_dynamics
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('yaml_path', metavar='path', type=str, help='path to yaml file')
args = parser.parse_args()

with open(args.yaml_path) as y:
    conf = yaml.load(y.read(), Loader=yaml.FullLoader)
run_dynamics(**conf["General setting"])







