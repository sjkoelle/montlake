# 🛶 Montlake
Welcome to Montlake.
Montlake contains tools for geometric data analysis.
It includes vector field group lasso and basis pursuit methods for parametric manifold learning.
It also contains differentiable shape featurizations including interpoint distances, planar angles, and torsions from data positions.
It uses embedding and tangent estimation components of the megaman package.

# Installation
You can install Montlake into Python directly from the command line without cloning the repository. We recommend the use of [Anaconda](www.anaconda.com).

```
conda create -n 'montlake' python=3.6
source activate montlake
conda install --channel=conda-forge -y pip nose coverage cython numpy seaborn scipy scikit-learn pyflann pyamg h5py plotly
pip install git+https://github.com/sjkoelle/montlake/
conda install nb_conda_kernels
python -m ipykernel install --user --name montlake --display-name "Python (montlake)”
```

# Troubleshooting
You may also need to install libmagic (e.g. ```brew install libmagic```), install pip (```conda install pip```), or install ipykernel (```pip install ipykernel```).
If you are still having issues installing using pip, clone the repository (```git clone git@github.com:sjkoelle/montlake.git```) and run

```
cd montlake
nbdev_build_lib
```

If there is a compatibility issue with `nb_conda_kernels`, which may result in failure of installation of jupyter notebook, you can also try the following to install Montlake.

```
conda create -n 'montlake' python=3.6
source activate montlake
conda install pip
pip install ipykernel
python -m ipykernel install --user --name montlake --display-name "Python (montlake)”
conda install --channel=conda-forge -y nose coverage cython numpy seaborn scipy scikit-learn pyflann pyamg h5py plotly
pip install git+https://github.com/sjkoelle/montlake/
```

# Usage

After installation, clone the repository (```git clone git@github.com:sjkoelle/montlake.git```).
Code from the experiments subfolder can be used recreate results from [Manifold Coordinates with Physical Meaning](https://arxiv.org/abs/1811.11891). 
Individual analyses are parameterized in experiments/configs and can be called from the command line.
Please set ROOT_DIR to your Montlake path and DATA_DIR to a directory containing the data (which can be downloaded [here](https://drive.google.com/drive/folders/1MKWF1k6X02K-BaQn4G-L_FjIZGcXPIn_?usp=sharing)). Please cite [this paper](https://www.nature.com/articles/s41467-018-06169-2) if you use this data.

Please use the following commands to set paths.

```
export ROOT_DIR=yourmontlakepath
export DATA_DIR=yourdatapath
```

Subsequently, you may configure the .json files with appropriate local file system paths and run the following commands in terminal to reproduce the computational and plotting analyses of synthetic and real data experiments. 

- Swiss Roll experiment:  
```jupyter notebook experiments/swiss_roll_resubmission.ipynb```

- Rigid Ethanol (no noise) Simulation: 
```
jupyter notebook experiments/generate_rigid_ethanol_data.ipynb
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/rigidethanol_diagram.json --outdir $DATA_DIR/rigidethanol --raw_data $DATA_DIR/raw_data/rigidethanol_nonoise.npy --nreps 25 --mflasso --name re_nonoise_diagram_mf
python -m montlake.exec.plotting --config $ROOT_DIR/experiments/configs_plotting/jmlr/reth_diagram_mf_jmlr.json
```

- Ethanol data with diagram dictionary
```
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_diagram.json --outdir $DATA_DIR/processed_data/eth_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_diagram_mflasso_122221
python -m montlake.exec.plotting --config $ROOT_DIR/experiments/configs_plotting/jmlr/eth_diagram_mf_jmlr.json
```

- Ethanol data with full dictionary
```
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/ethanol_full.json --outdir $DATA_DIR/processed_data/eth_full_mflasso_122221 --raw_data $DATA_DIR/raw_data/ethanol.mat --nreps 25 --mflasso --name eth_full_mflasso_122221
python -m montlake.exec.plotting --config $ROOT_DIR/experiments/configs_plotting/jmlr/eth_full_mf_jmlr.json
```

- Malonaldehyde data with diagram dictionary
```
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_diagram.json --outdir $DATA_DIR/processed_data/mal_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_diagram_mflasso_122221
python -m montlake.exec.plotting --config $ROOT_DIR/experiments/configs_plotting/jmlr/mal_diagram_mf_jmlr.json
```

- Malonaldehyde data with full dictionary
```
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/malonaldehyde_full.json --outdir $DATA_DIR/processed_data/mal_full_mflasso_122221 --raw_data $DATA_DIR/raw_data/malonaldehyde.mat --nreps 25 --mflasso --name mal_full_mflasso_122221
python -m montlake.exec.plotting --config $ROOT_DIR/experiments/configs_plotting/jmlr/mal_full_mf_jmlr.json
```
- Toluene data with diagram dictionary
```
python -u -m montlake.exec.run_exp --config $ROOT_DIR/experiments/configs/toluene_diagram.json --outdir $DATA_DIR/processed_data/tol_diagram_mflasso_122221 --raw_data $DATA_DIR/raw_data/toluene.mat --nreps 25 --mflasso --name tol_diagram_mflasso_122221
python -m montlake.exec.plotting --config $ROOT_DIR/experiments/configs_plotting/jmlr/tol_diagram_mf_jmlr.json
```

Code for the Tangent Space Lasso is also colocated in this repo.

# Contributing

Please feel free to contribute, branch, fork, or edit.
This package was built using nbdev, and so source code is kept in the nbs folder and then compiled by running nbdev_build_lib.
Please cite if you use!
