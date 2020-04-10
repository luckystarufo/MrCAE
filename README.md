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
            |- torch_cae_multilevel_V4.py (*this is our MrCAE torch code*)
	    |- torch_cae_skip_connection.py (classical benchmark)
	  |- scripts/
	    |- MrCAE_{xxxx}.ipynb (individual experiments)
	    |- benchmark_{xxxx}.ipynb (benchmark experiments)
	    |- training/ (contains scripts for batch training on clusters)
      	  |- data/ (contains simulation data, fluids data, SST data)
      	  |- model/(contains models at different stages)
      	  |- result/(contains training results)

## Introduction
- This repo provides the code for the paper "Multiresolution Convolutional Autoencoders" by Yuying Liu, Colin Ponce, Steven L. Brunton and J. Nathan Kutz (in review). 
- The network architecture is recursively built up to process data across different resolutions — architectures built for processing coarser data are later embedded into the next-level architectures to ensure knowledge transfer. 

![figure 1: architecture overview](./figures/MrCAE_overview.jpeg?raw=true)

- Within each level, we perform one *deepening* operation and a sequence of *widening* operations. The deepening operation inserts a convolutional/deconvolutional layer between the current and previous level inputs/outputs. This is the transfer learning step. The widening operation expands the network capacity in order to capture the new, finer-grained features of the higher resolution data. 

![figure 2: deepening & widening operations](./figures/MrCAE_overview2.jpeg?raw=true)

- The convolutional filters we use are highly *adaptive*: they only refine the regions that produce large reconstructions errors.

![figure 3: adaptive filters](./figures/MrCAE_overview3.jpeg?raw=true)

- By utilizeing the above features, our network can progressively 'grow' itself and end up with the 'right' amount of parameters for characterzing the data. Different level of performances can be achieved at different stages of training.

![figure 4: intermediate results](./figures/reconstructions.gif?raw=true)


## Usage
- We provide three methods to train the architecture: train\_net(), train\_net\_one\_level() and train\_net\_one\_stage() which corresponds to (i) end-to-end training, (ii) train one level only and (iii) train one stage only (after an architecture change). 
- See demos under scripts/ for more details
- Notes: training on GPUs is highly recommended.

## Citation
```
TBD
```
