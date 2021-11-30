#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=largemem
#SBATCH --mem=10000
#SBATCH --job-name=e_f_ts_500
#SBATCH --error=e_f_ts_500.err
#SBATCH --output=e_f_ts_500.log
#SBATCH --mail-type=FAIL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
source activate montlake
cd ~/montlake
python -u -m montlake.exec.run_exp --config /homes/sjkoelle/montlake/experiments/configs/ethanol_full.json --nsel 500 --outdir /homes/sjkoelle/thesis_data/processed_data_2/ethanol_500 --raw_data /homes/sjkoelle/thesis_data/raw_data/ethanol.mat --nreps 25 --tslasso --name eth_full_ts_500
source deactivate
echo "end"
