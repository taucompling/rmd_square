# Replicator-Mutator-Dynamics
Original paper: https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12681

Game-theoretic model of competition between types of language users, each endowed with certain lexical representations and a particular pragmatic disposition to act on them. 

## Setup for usage

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
### Getting started

For running the experiments, cd into the project folder `RMD` and execute `main.py` + `path_to_yaml`:
```bash
python3 main.py path_to_yaml.yaml
```
For example:
```bash
python3 main.py configs/3x3_bb.yaml
```
There are predefined yamls in the folder `configs`. They are of the form "messages x states_cost.yaml". For examlpe "3x3_bb.yaml" contains the configuration for an experiment with lexica of size 3x3 (3 states and 3 messages) and the building-block cost function. 
Use one of the predefined yamls or copy one and adjust it for your needs.

The results get stored in a folder `experiments`.

CONTENTS
------------
The folder structure is as follows:
```
RMD/
├── configs
│   ├── 3x3_3x4_bb.yaml
│   ├── 3x3_3x4_bh.yaml
│   ├── 3x3_bb.yaml
│   ├── 3x3_bh.yaml
│   ├── 3x4_bb.yaml
│   ├── 3x4_bh.yaml
│   └── z_predefined
│       ├── 3x3_3x4_predefined.yaml
│       ├── 3x3_predefined.yaml
│       └── 3x4_predefined.yaml
├── main.py
├── predefined_lexica
│   ├── 3x3.txt
│   ├── 3x4.txt
│   └── comb.txt
├── py_scripts
│   ├── __init__.py
│   ├── lexica.py
│   ├── mutation_matrix.py
│   ├── mutual_utility.py
│   ├── player.py
│   ├── plots
│   │   ├── __init__.py
│   │   ├── plot_progress.py
│   │   ├── pragmatic_vs_literal.py
│   │   ├── proportion_target_types.py
│   │   └── x_best_prag_lit.py
│   └── rmd.py
├── README.md
└── requirements.txt




