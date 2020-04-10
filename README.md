**MrCAE**: multiresolution convolutional autoencoders (with adaptive filters)

## Table of contents
* [Structure](#structure)
* [Introduction](#introduction)
* [Usage](#usage)
* [Citation](#citation)


## Structure
    MrCAE/
      |- README.md
      |- src/
         |- __init__.py
         |- utils.py
         |- torch_cae_multilevel_V4.py (this is our MrCAE torch code)
	 |- torch_cae_skip_connection.py (classical benchmark)
      |- scripts/
         |- MrCAE_{xxxx}.ipynb (individual experiments)
	 |- benchmark_{xxxx}.ipynb (benchmark experiments)
	 |- training/ (contains scripts for batch training on clusters)
      |- data/ (contains simulation data, fluids data, SST data)
      |- model/(contains models at different stages)
      |- result/(contains training results)

## Introduction
This repo provides the code for the paper "Multiresolution Convolutional Autoencoders" by Yuying Liu, Colin Ponce, Steven L. Brunton and J. Nathan Kutz (in review). 

## Usage
- Basically, we provide three methods to train the architecture: train\_net(), train\_net\_one\_level() and train\_net\_one\_stage() which corresponds to (i) end-to-end training, (ii) train one level only and (iii) train one stage only (after an architecture change). 
- See demos under scripts/
- Notes: training on GPUs is highly recommended.

## Citation
```
TBD
```
