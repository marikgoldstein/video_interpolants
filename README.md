# Summary

This is a repo for conditional (semi-markovian) video modeling using generative models (specifically flow matching / stochastic interpolants).

The original task studied here was defined by [RIVER]((https://github.com/Araachie/river). They conditionally model frame T+1 given frame T as well as a randomly chosen frame between 1 and T-1. Moreover, to save compute, the generative modeling is done in the latent space of a pre-trained VQVAE.

Mark and collaborates then adapted this code to study the choice of interpolant 
and how it affects results (not so much to study the overall video modeling technique). We did this in the paper

[Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes](https://arxiv.org/abs/2403.13724).

The code here has been re-written from the RIVER repo to make something more shareable and with fewer files. As such, certain extra functionality from that repo is missing. This repo focuses on the basics of training a model and sampling from it.

data: Mark is still working on reproducing the scripts for data modeling. This was made complicated by some of the original data sources dissapearing. 
Meanwhile, just ask mark for the preprocessed HDF5 shards, which are all that are needed for training/sampling. Mark will update when the original data processing is available.

vqvae checkpoints: Ask Mark also for the KTH and CLEVRER pre-trained VQVAEs, or [download them from the RIVER repo](https://github.com/Araachie/river?tab=readme-ov-file#pretrained-models). They were not changed for this project.

Assuming you have the data shards (necessary) and the vqvae checkpoints (necessary), and optionally the model checkpoints from the Follmer Processes paper, 
the next step is to run the modeling code.

# some dependencies 

- wandb for logging
- pip install albumentations, for data augmentation 
- pip install moviepy imageio, for being able to log videos to wandb
- if building data from scratch, check that you can use ffmpeg. For me on the cluster, it is "module load ffmpeg"

# Running the code

check out main.py and configs.py. Main.py just exposes a few basic arguments
- which dataset
- which interpolant
- whether or not to do an overfitting/debugging session (use 0 or 1 and they are converted to False or True)
- whether to load the model from a model checkpoint (this is just for the main model, the VQVAE is automatically loaded)
- wandb entity (username) and project name

The configs file then adds on many things. There is a shared section applicable to both datasets, and then separate branches for KTH vs CLEVRER. Make sure to edit
- the location of the data shards
- the location of the vqvae checkpoints
- the location where you would like to save results

Then, run main.py with appropriate args. This should call train.py and the train loop should start.


# data processing (WORK IN PROGRESS, IGNORE)

This is made complicated in part by the KTH data website having recently broken links to the data zips. Mark will transfer data
from compute cluster to some public place and give instructions. The data processing code will be based on a mix of
- RIVER (https://github.com/Araachie/river). 
- https://github.com/willi-menapace/PlayableVideoGeneration 
- https://github.com/edouardelasalles/srvp/. 

Meanwhile, Mark will just share the preprocessed HDF5 shards for KTH and CLEVRER with anyone that wants to run this code.

- once you have kth_data/raw/jogging etc which contains AVIs, do python preprocessing avis_to_pngs.py
- then split the png dirs into train val test. I put videos d1,d2 in train, d3 in val, and d4 in test for this demo repo.
- data/
	- train/
		- 00000/ <- this is one video
		- 00001/
			- 000.png <- these are frames within one video
			- 001.png
			- 002.png
	- val/
	- test/
- then do python pngs_to_hdf5_shards.py --out_dir kth_data/hdf5s --data_dir kth_data/processed --image_size 64 --extension png
- download autoencders and optionally model checkpoints from river repo https://github.com/Araachie/river

--ckpt_fname ckpts/river_kth_64_model.pth 

- their evaluation code isn’t great + not all pieces are included. 

# FVD

FVD wasn’t included. I found a separate repo to compute that.  https://github.com/google-research/google-research/blob/20b2520e416edaea8c038bbf54cc1c739c542822/frechet_video_distance/README.md
