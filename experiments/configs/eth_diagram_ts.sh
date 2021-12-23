#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=short
#SBATCH --mem=10000
#SBATCH --job-name=eth_diag_ts
#SBATCH --error=eth_diag_ts.err
#SBATCH --output=eth_diag_ts.log
#SBATCH --mail-type=FAIL       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sjkoelle@gmail.com # Email to which notifications will be sent

export PATH="~/anaconda3/bin:$PATH"
export ROOT_DIR=/homes/sjkoelle/montlake
export DATA_DIR=/homes/sjkoelle/thesis_data

source activate montlake
cd ~/montlake
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_diagram.json --outdir $DATA_DIR/processed_data/eth_diagram_tslasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --tslasso --name eth_diagram_tslasso_122221
source deactivate
echo "end"