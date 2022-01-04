# Montlake 🛶
Welcome to Montlake.
Montlake contains tools for geometric data analysis.
It includes vector field group lasso and basis pursuit methods for parametric manifold learning.
It also contains differentiable shape featurizations including interpoint distances, planar angles, and torsions from data positions.
It uses embedding and tangent estimation components of the megaman package.

# Installation
You can install Montlake into Python directly from the command line without cloning the repository. 

```
conda create -n 'montlake' python=3.6
source activate montlake
conda install --channel=conda-forge -y pip nose coverage cython numpy scipy scikit-learn pyflann pyamg h5py plotly
pip install git+git://github.com/sjkoelle/montlake/
```

# Usage

After installation, code from the experiments folder can be used recreate results from Manifold Coordinates with Physical Meaning (https://arxiv.org/abs/1811.11891).
Individual analyses are parameterized in experiments/configs and can be called from the command line.

```
python -u -m montlake.exec.run_exp --config montlake/experiments/configs/rigidethanol_diagram.json --outdir data/rigidethanol --raw_data data/rigidethanol_nonoise.npy --nreps 1 --mflasso --name re_nonoise_diagram_mf
```

Molecules:
```
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_diagram.json --outdir $DATA_DIR/processed_data/eth_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_diagram_mflasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_diagram.json --outdir $DATA_DIR/processed_data/mal_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_diagram_mflasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_full.json --outdir $DATA_DIR/processed_data/mal_full_mflasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_full_mflasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_full.json --outdir $DATA_DIR/processed_data/eth_full_mflasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_full_mflasso_122221
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/toluene_diagram.json --outdir $DATA_DIR/processed_data/tol_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/toluene.mat --nreps 25 --mflasso --name tol_diagram_mflasso_122221

python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/eth_full_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/eth_diagram_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/tol_diagram_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/mal_full_mf_jmlr.json
python -m montlake.exec.plotting --config /Users/samsonkoelle/tunatostada/experiments/configs_plotting/jmlr/mal_diagram_mf_jmlr.json

```

This will save a dictionary with relevant experimental results.  These results are plotted in experiments/nbs.

# Contributing

Please feel free to contribute, branch, fork, or edit.
This package was built using nbdev, and so source code is kept in the nbs folder and then compiled by running nbdev_build_lib.
Please cite if you use!
