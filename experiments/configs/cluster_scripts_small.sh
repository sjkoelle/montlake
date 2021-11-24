#!/bin/bash
#SBATCH --job-name montlake      # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition largemem     # Slurm partition to use
#SBATCH --ntasks 16          # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-24:00        # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=40000     # Memory limit for each tasks (in MB)
#SBATCH -o myscript_montlake%j.out    # File to which STDOUT will be written
#SBATCH -e myscript_montlake%j.err    # File to which STDERR will be written
#SBATCH --mail-type=ALL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate montlake
python -u -m montlake.exec.run_exp --config /homes/sjkoelle/montlake/experiments/configs/rigidethanol_diagram.json --outdir /homes/sjkoelle/thesis_data/processed_data/rigidethanol --raw_data /homes/sjkoelle/thesis_data/raw_data/rigidethanol_nonoise.npy
