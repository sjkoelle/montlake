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
export ROOT_DIR=/homes/sjkoelle/montlake
export DATA_DIR=/homes/sjkoelle/thesis_data
source activate montlake
cd ~/montlake
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_diagram.json --outdir $DATA_DIR/processed_data/mal_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_diagram
source deactivate
echo "end"
