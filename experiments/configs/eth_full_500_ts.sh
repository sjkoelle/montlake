#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=largemem
#SBATCH --mem=10000
#SBATCH --job-name=eth_full_ts
#SBATCH --error=eth_full_ts.err
#SBATCH --output=eth_full_ts.log
#SBATCH --mail-type=FAIL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate montlake
cd ~/montlake
python -u -m montlake.exec.run_exp --nsel 500 --config /homes/sjkoelle/montlake/experiments/configs/ethanol_full.json --outdir /homes/sjkoelle/thesis_data/processed_data_2/ethanol --raw_data /homes/sjkoelle/thesis_data/raw_data/ethanol.mat --nreps 1 --tslasso --name eth_full_ts
source deactivate
echo "end"
