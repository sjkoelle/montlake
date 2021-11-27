#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --mem=10000
#SBATCH --job-name=mal_diagram_mf
#SBATCH --error=mal_diagram_mf.err
#SBATCH --output=mal_diagram_mf.log
#SBATCH --mail-type=FAIL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate montlake
cd ~/montlake
python -u -m montlake.exec.run_exp --config /homes/sjkoelle/montlake/experiments/configs/malonaldehyde_diagram.json --outdir /homes/sjkoelle/thesis_data/processed_data_2/malonaldehyde --raw_data /homes/sjkoelle/thesis_data/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_diagram_mf
source deactivate
echo "end"