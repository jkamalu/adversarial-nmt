#!/bin/bash
#SBATCH --partition=jag-standard
#SBATCH --nodes=1
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:titanx:1

#SBATCH --job-name="bert-vanilla-hidden"
#SBATCH --output=sample-%j.out

# only use the following if you want email notification

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
srun --nodes=${SLURM_NNODES} bash -c 'source /nlp/u/jkamalu/miniconda3/bin/activate && conda activate adversarial-nmt && python run.py --mode train --experiment bert-vanilla-hidden.yml'

# done
echo "Done"
