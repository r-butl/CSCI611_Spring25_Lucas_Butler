# Prerequisites

- shell with miniconda/anaconda installed, jupyter notebook software
- Recommended: I use VSCode and the Jupyter extension developed by Microsoft. This takes care of the Jupyter configuration and discovery of the conda environment for me. I don't know what the installation/setup is for folks who don't use VSCode as their editor. I prefer it because of the useful Jupyter integration.

# Getting started

To run the 'image_filtering.ipynb' and 'build_cnn.ipynb' notebooks with my implementations of the CNN and image filtering pipeline, first install the necessary packages using miniconda and the provided environment.yml file. Run the following commands to install the environment:

```
conda create -n ml -f environment.yml
```

```
conda activate ml
```

After doing this, open the jupyter notebooks and select the 'ml' kernel, then run the code.
