# Bi-directional NMT with Adversarial Regularization

## Installation

We use Anaconda virtual environments to manage dependencies. Install Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and activate the base environment. Then create and activate a new virtual environment with Python 3.6
1. `source /path/to/conda/bin/activate`
2. `conda create -n adversarial-nmt python=3.6`
3. `conda activate adversarial-nmt`

Install all depencies with conda and pip. Ensure `pip` points to the corresponding virtual environment binary. 
1. `conda install pytorch torchvision cudatoolkit=XX.X -c pytorch` where `XX.X` will depend on your CUDA installation
2. `which pip`
3. `pip install -r requirements.txt`
4. `git clone https://github.com/pytorch/fairseq`
5. `cd fairseq && pip install -e ./`

## Data

We use the europarl-v7 English-French parallel corpus. Download and unzip the data.

#### Download and unzip
1. `cd data && wget http://statmt.org/europarl/v7/fr-en.tgz && tar -xzvf fr-en.tgz`

## Training and Evaluation with `run.py`
We provide a Python script for training and evaluation on Europarl. All experiment parameters are entirely defined by a user provided YAML config file. Execute `python run.py --help` for further instruction.

## Jupyter

You may wish to use Jupyter notebooks for development and visualization. Launch a Jupyter server and open `Run.ipynb`. Ensure `jupyter` points to the corresponding virtual environment binary. 

1. `which jupyter`
2. `jupyter notebook --ip=0.0.0.0 --port=XXXX --no-browser &` where `XXXX` corresponds to an exposed port on your machine. For SAIL, we can use `8880`.
