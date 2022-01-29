# RMD_Square

The typology of lexicalizations in natural languages is highly skewed: some meanings repeatedly receive their own expression as individual morphemes or words in language after language, while many other meanings rarely or never do. For example, while many languages have monomorphemic counterparts of English *some* and *all*, no known language has a monomorphemic quantifier that means *all or none* or a quantifier that asserts that its two arguments are of the same cardinality. It seems tempting to reason from this typological skew to properties of stored representations. However, it is not generally safe to assume that if something is typologically unattested then it simply cannot be represented or learned. The representational system for stored denotations is just one of several interacting factors that affect the typology, and other factors such as communicative pressure and learnability are likely to shape patterns of lexicalization. In this paper we propose to reason from the typology to stored representations by modeling the representational framework, communicative pressure, and learnability directly within an evolutionary model, building on work by Brochhagen, Franke at al (https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12681). Our empirical focus is a lexicalization asymmetry noted by Horn (1972) in the domain of logical operators and framed within the Aristotelian Square of Opposition. We show that, on certain assumptions, Horn's lexicalization pattern depends on very particular representational costs in the lexicon: it arises if the storage costs for *every* and *some* are lower than those for *not every* and *not some* but not otherwise. 


## Setup for usage
------------
### Cloning the project

In order to run tests, clone the repository and cd into the
`rmd_square` folder. 

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
│   │   ├── __init__.py
│   │   ├── plot_progress.py
│   │   ├── proportion_target_types.py
│   │   └── x_best_prag_lit.py
│   ├── rmd.py
│   ├── uegaki_utility.py
│   └── utils.py
├── README.md
└── requirements.txt





