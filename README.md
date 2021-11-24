# montlake 🛶
Welcome to montlake: tools for geometric data analysis.
Montlake contains group lasso and basis pursuit algorithms for parametric manifold learning and interfacing code for computing interpoint distances, planar angles, and torsions.  It uses embedding and geometry components of the megaman package.

# Installation

```
conda create -n 'montlake' python=3.6
conda activate montlake
conda install --channel=conda-forge -y pip nose coverage cython numpy scipy scikit-learn pyflann pyamg h5py plotly
pip install git+git://github.com/sjkoelle/montlake/
```

# Usage

Code from the experiments folder can be used recreate results from Manifold Coordinates with Physical Meaning (https://arxiv.org/abs/1811.11891).
Individual analyses are parameterized in experiments/configs and can be called from the command line.

```
python -m montlake.exec.run_exp --config /Users/samsonkoelle/tunatostada/experiments/configs/malonaldehyde_full.json
```

This will save a dictionary with relevant experimental results.  These results are plotted in experiments/nbs.

# Contributing

Please feel free to contribute, branch, fork, or edit.
This package was built using nbdev, and so source code is kept in the nbs folder and then compiled by running nbdev_build_lib.
