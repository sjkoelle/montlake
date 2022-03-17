export ROOT_DIR=/homes/sjkoelle/montlake
export DATA_DIR=/homes/sjkoelle/thesis_data

#finalized
#wrong names
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_diagram.json --outdir $DATA_DIR/processed_data/eth_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_diagram_mflasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_diagram.json --outdir $DATA_DIR/processed_data/mal_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_diagram_mflasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_full.json --outdir $DATA_DIR/processed_data/mal_full_mflasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_full_mflasso_122221
#right names
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_full.json --outdir $DATA_DIR/processed_data/eth_full_mflasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_full_mflasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/toluene_diagram.json --outdir $DATA_DIR/processed_data/tol_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/toluene.mat --nreps 25 --mflasso --name tol_diagram_mflasso_122221

python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_diagram.json --outdir $DATA_DIR/processed_data/mal_diagram_tslasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --tslasso --name mal_diagram_tslasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_full.json --outdir $DATA_DIR/processed_data/mal_full_tslasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --tslasso --name mal_full_tslasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_diagram.json --outdir $DATA_DIR/processed_data/eth_diagram_tslasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --tslasso --name eth_diagram_tslasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_full.json --outdir $DATA_DIR/processed_data/eth_full_tslasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --tslasso --name eth_full_tslasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/toluene_diagram.json --outdir $DATA_DIR/processed_data/tol_diagram_tslasso_122221 --raw_data $DATA_DIR/raw_data/toluene.mat --nreps 25 --tslasso --name tol_diagram_tslasso_122221

python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_full.json --outdir $DATA_DIR/processed_data/mal_full_tslasso500_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --nsel 500 --tslasso --name mal_full_tslasso500_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_full.json --outdir $DATA_DIR/processed_data/eth_full_tslasso500_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --nsel 500 --tslasso --name eth_full_tslasso500_122221

#downloaded
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/mal_full_tslasso500_122221 ~/thesis_data/processed_data/malonaldehyde
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/mal_diagram_tslasso_122221 ~/thesis_data/processed_data/malonaldehyde
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/eth_full_tslasso500_122221 ~/thesis_data/processed_data/ethanol
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/mal_full_tslasso_122221 ~/thesis_data/processed_data/malonaldehyde
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/eth_full_tslasso_122221 ~/thesis_data/processed_data/ethanol
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/mal_full_mflasso_122221 ~/thesis_data/processed_data/malonaldehyde
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/mal_diagram_tslasso_122221 ~/thesis_data/processed_data/malonaldehyde
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/mal_diagram_mflasso_122221 ~/thesis_data/processed_data/malonaldehyde
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/eth_full_mflasso_122221 ~/thesis_data/processed_data/ethanol
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/eth_diagram_mflasso_122221 ~/thesis_data/processed_data/ethanol
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/tol_diagram_mflasso_122221 ~/thesis_data/processed_data/toluene
scp -r sjkoelle@warthog.stat.washington.edu:~/thesis_data/processed_data/tol_diagram_tslasso_122221 ~/thesis_data/processed_data/toluene

#downloading
#ran

python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/eth_full_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/eth_diagram_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/tol_diagram_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/mal_full_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/mal_diagram_mf_jmlr.json


python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/eth_full_ts_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/eth_diagram_ts_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/tol_diagram_ts_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/mal_full_ts_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/mal_diagram_ts_jmlr.json


#unran
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/eth_full_mf.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/eth_diagram_mf.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/tol_diagram_mf.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/mal_full_mf.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/mal_diagram_mf.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/tol_diagram_ts.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/eth_diagram_ts.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/mal_full_ts.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/mal_diagram_ts.json
ython -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/eth_full_ts.json

python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/eth_full_500_ts.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/mal_full_500_ts.json




