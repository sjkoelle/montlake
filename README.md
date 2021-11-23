# montlake 🛶
Welcome to the Montlake package.
This package includes a variety of tools for geometric data analysis.

# Installation

```
conda create -n 'montlake' python=3.6
conda activate montlake
conda install --channel=conda-forge -y pip nose coverage cython numpy scipy scikit-learn pyflann pyamg h5py plotly
pip install git+git://github.com/sjkoelle/montlake/
```

# Usage

Code from this package can be used recreate results from Manifold Coordinates with Physical Meaning (https://arxiv.org/abs/1811.11891).
