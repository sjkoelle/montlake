# montlake 🛶
Welcome to Montlake: tools for geometric data analysis.
This package was built using nbdev. 
It contains group lasso and basis pursuit algorithms for parametric manifold learning and interfacing code for computing interpoint distances, planar angles, and torsions.

# Installation

```
conda create -n 'montlake' python=3.6
conda activate montlake
conda install --channel=conda-forge -y pip nose coverage cython numpy scipy scikit-learn pyflann pyamg h5py plotly
pip install git+git://github.com/sjkoelle/montlake/
```

# Usage

Code from the experiments folder can be used recreate results from Manifold Coordinates with Physical Meaning (https://arxiv.org/abs/1811.11891).
