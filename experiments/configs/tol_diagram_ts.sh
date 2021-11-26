#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --mem=10000
#SBATCH --job-name=tol_ts
#SBATCH --error=tol_ts.err
#SBATCH --output=tol_mf.log
#SBATCH --mail-type=FAIL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate montlake
cd ~/montlake
python -u -m montlake.exec.run_exp --config /homes/sjkoelle/montlake/experiments/configs/toluene_diagram.json --outdir /homes/sjkoelle/thesis_data/processed_data_2/toluene --raw_data /homes/sjkoelle/thesis_data/raw_data/toluene.mat --nreps 25 --tslasso --name tol_diagram_ts
source deactivate
echo "end"