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

We use the europarl-v7 English-French parallel corpus. Download, unzip, shuffle, and split the data.

#### Download and unzip
1. `cd data && wget http://statmt.org/europarl/v7/fr-en.tgz && tar -xzvf fr-en.tgz`
#### Shuffle
2. `shuf --random-source=europarl-v7.fr-en.en europarl-v7.fr-en.en > europarl-v7.fr-en.en.shuf` 
3. `shuf --random-source=europarl-v7.fr-en.en europarl-v7.fr-en.fr > europarl-v7.fr-en.fr.shuf`
#### Split 
4. `EUROPARL_LINES=$(cat europarl-v7.fr-en.en | wc -l)`
5. `head -$((EUROPARL_LINES-100000)) europarl-v7.fr-en.en.shuf > europarl-v7.fr-en.en.train`
6. `head -$((EUROPARL_LINES-100000)) europarl-v7.fr-en.fr.shuf > europarl-v7.fr-en.fr.train`
7. `tail -100000 europarl-v7.fr-en.en.shuf > europarl-v7.fr-en.en.val`
8. `tail -100000 europarl-v7.fr-en.fr.shuf > europarl-v7.fr-en.fr.val`

## Jupyter

We use Jupyter notebooks for development and visualization (subject to change). Launch a Jupyter server and open `Run.ipynb`. Ensure `jupyter` points to the corresponding virtual environment binary. 

1. `which jupyter`
2. `jupyter notebook --ip=0.0.0.0 --port=XXXX --no-browser &` where `XXXX` corresponds to an exposed port on your machine. For SAIL, we can use `8880`.
