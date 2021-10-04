# Replicator-Mutator-Dynamics
Original paper: https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12681

Game-theoretic model of competition between types of language users, each endowed with certain lexical representations and a particular pragmatic disposition to act on them. 

Includes extensions like other cost functions, different inventories, different calculation of mutual utility, grid searches etc.

## Setup for usage
------------
### Cloning the project

In order to run tests, clone the repository and cd into the
`RMD` folder. 

### Installation

cd into the root directory of the project, create a virtual environment and install the requirements.

```bash
cd RMD
python3 -m venv env
source env/bin/activate 
pip3 install -r requirements.txt
```
## Getting started
------------
For running the experiments, cd into the project folder `RMD` and execute `main.py` + `path_to_yaml`:
```bash
python3 main.py path_to_yaml.yaml
```
For example:
```bash
python3 main.py configs/example.yaml
```
You can find the example file `example.yaml` in the `configs` folder. Use this one or copy it and adjust it for your needs.

The results get stored in a folder (name specified in the yaml).

You can define your own cost functions if you set `cost` in the yaml to new_approach and adjust the cost function in `message_cost.py` at new_approach.

## Contents
------------
The folder structure is as follows:
```
RMD
├── configs
│   └── example.yaml
├── irrelevant
│   └── process_predefined_lexica.py
├── main.py
├── py_scripts
│   ├── checks.py
│   ├── __init__.py
│   ├── lexica.py
│   ├── message_costs.py
│   ├── mutation_matrix.py
│   ├── mutual_utility.py
│   ├── player.py
│   ├── plots
│   │   ├── informativesness_score.py
│   │   ├── __init__.py
│   │   ├── plot_all_sizes.py
│   │   ├── plot_progress.py
│   │   ├── proportion_target_types.py
│   │   └── x_best_prag_lit.py
│   ├── rmd.py
│   ├── uegaki_utility.py
│   └── utils.py
├── README.md
└── requirements.txt





