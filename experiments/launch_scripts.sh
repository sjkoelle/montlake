export ROOT_DIR=/homes/sjkoelle/montlake
export DATA_DIR=/homes/sjkoelle/thesis_data

#finalized
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/ethanol_diagram.json --outdir {DATA_DIR}/processed_data/eth_diagram_mflasso_122221 --raw_data {DATA_DIR}/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_diagram


#not finalized
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/malonaldehyde_diagram.json --outdir {DATA_DIR}/processed_data/mal_diagram_mflasso_122221 --raw_data {DATA_DIR}/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_diagram
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/malonaldehyde_full.json --outdir {DATA_DIR}/processed_data/mal_full_mflasso_122221 --raw_data {DATA_DIR}/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_full
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/ethanol_full.json --outdir {DATA_DIR}/processed_data/eth_full_mflasso_122221 --raw_data {DATA_DIR}/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_full
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/toluene_diagram.json --outdir {DATA_DIR}/processed_data/tol_diagram_mflasso_122221 --raw_data {DATA_DIR}/raw_data/toluene.mat --nreps 25 --mflasso --name tol_diagram

python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/malonaldehyde_diagram.json --outdir {DATA_DIR}/processed_data/mal_diagram_tslasso_122221 --raw_data {DATA_DIR}/raw_data/malonaldehyde.mat --nreps 25 --tslasso --name mal_diagram
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/malonaldehyde_full.json --outdir {DATA_DIR}/processed_data/mal_full_tslasso_122221 --raw_data {DATA_DIR}/raw_data/malonaldehyde.mat --nreps 25 --tslasso --name mal_full
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/ethanol_diagram.json --outdir {DATA_DIR}/processed_data/eth_diagram_tslasso_122221 --raw_data {DATA_DIR}/raw_data/ethanol.mat --nreps 25 --tslasso --name eth_diagram
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/ethanol_full.json --outdir {DATA_DIR}/processed_data/eth_full_tslasso_122221 --raw_data {DATA_DIR}/raw_data/ethanol.mat --nreps 25 --tslasso --name eth_full
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/toluene_diagram.json --outdir {DATA_DIR}/processed_data/tol_diagram_tslasso_122221 --raw_data {DATA_DIR}/raw_data/toluene.mat --nreps 25 --tslasso --name tol_diagram

python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/malonaldehyde_full.json --outdir {DATA_DIR}/processed_data/mal_full_tslasso_122221 --raw_data {DATA_DIR}/raw_data/malonaldehyde.mat --nreps 25 --tslasso --name mal_full
python -u -m montlake.exec.run_exp --config {ROOT_DIR}/experiments/configs/ethanol_full.json --outdir {DATA_DIR}/processed_data/eth_full_tslasso_122221 --raw_data {DATA_DIR}/raw_data/ethanol.mat --nreps 25 --tslasso --name eth_full

