#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=largemem
#SBATCH --mem=50000
#SBATCH --job-name=e_f_ts_500
#SBATCH --error=e_f_ts_500.err
#SBATCH --output=e_f_ts_500.log
#SBATCH --mail-type=FAIL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
export ROOT_DIR=/homes/sjkoelle/montlake
export DATA_DIR=/homes/sjkoelle/thesis_data

source activate montlake
cd ~/montlake
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_full.json --outdir $DATA_DIR/processed_data/eth_full_tslasso500_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --nsel 500 --tslasso --name eth_full_tslasso500_122221
source deactivate
echo "end"
