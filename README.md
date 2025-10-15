# Summary

This is a repo for conditional (semi-markovian) video modeling using stochastic interpolants, from the paper

[Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes](https://arxiv.org/abs/2403.13724)

Work done by Yifan Chen, Mark Goldstein, Mengjian Hua, Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden.

Repo maintained by MG and co-authors.

The original task studied here was defined by [RIVER](https://github.com/Araachie/river). RIVER conditionally model sframe T+1 given frame T as well as a randomly chosen frame between 1 and T-1. The generative modeling is done in the latent space of a pre-trained VQVAE. The authors of this work adapted that code to study the choice of interpolant and how it affects results (not so much to study the overall video modeling technique or the VQGANs).

A few notes:

- this is a more-readable re-write of the actual project code. MG has verified on some simple overfitting tests (on a datapoint or a batch). But MG will soon re-run full experiments to make sure the re-write was OK. Just reach out if any questions to goldstein AT nyu DOT edu or mgoldstein AT flatironstitute DOT org

- data: We study two datasets for video modeling, [KTH](https://www.csc.kth.se/cvap/actions/) and [CLEVRER](http://clevrer.csail.mit.edu/). The preprocessed HDF5 shards for running the KTH experiments can be [found here on Globus](https://app.globus.org/file-manager/collections/41785d4d-0395-41d7-80d5-c35c46396c95/overview). MG is working on also uploading the CLEVRER shards and the data pre-processing code (drafts of preprocessing code in work_in_progress subdir). But for now the code can just be run with the shards. 
MG will also uploaded the raw KTH data because its zip file links seem to have recently broken on the original KTH website! The CLEVRER links are alive and well, so we will likely not upload the large raw CLEVRER data.

- checkpoints: we host the [checkpoints here on Globus](https://app.globus.org/file-manager/collections/1b49bb33-ce78-4dd8-bb0d-bc5736d0ce18/overview). The VQVAE checkpoints come from [RIVER](https://github.com/Araachie/river?tab=readme-ov-file#pretrained-models) 
(thanks so much to them for open sourcing). The VQVAE checkpoints were not changed at all for this project, but we are just uploading for availability/redundancy. But the flow model checkpoints were trained by us for this project: one for each of two datasets and one for each choice of interpolant ("linear" and "ours").

Assuming you have the data shards (necessary) and the vqvae checkpoints (necessary), and optionally the model checkpoints, the next step is to run the modeling code.

# some dependencies 

- wandb for logging
- pip install albumentations, for data augmentation 
- pip install moviepy imageio, for being able to log videos to wandb
- if building data from scratch, check that you can use ffmpeg. For me on the cluster, it is "module load ffmpeg"

# Running the code

check out main.py and configs.py. Main.py just exposes a few basic arguments
- which dataset
- which interpolant
- whether or not to do an overfitting/debugging session (overfit arg can be "none" for regular experiment, "batch" for overfitting on a batch, or "one" for overfitting on a batch of one repeated datapoint). When overfitting, sampling always uses the overfitting batch for initial frames.
- whether or not to do a smoke test which sets all settings so that the script finishes quickly (about 100 steps) and you can identify that it loads and finishes without crashing while making sure that sampling, logging, and checkpointing all work. Use the ints 1 or 0 for True or False, and the code will convert it to a bool
- whether to load the model from a model checkpoint (this is just for the main model, the VQVAE is automatically loaded)
- wandb entity (username) and project name


The configs file then adds on many things. There is a shared section applicable to both datasets, and then separate branches for KTH vs CLEVRER. Make sure to edit
- the location of the data shards
- the location of the vqvae checkpoints
- the location where you would like to save results

Then, run main.py with appropriate args. This should call train.py and the train loop should start.

For an example, see run.sh. YOu can 

Some DDP systems use 

```
torchrun --standalone --nnodes=1 --nproc_per_node=${GPUS}
```

while others use the older

```
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=${GPUS}"
```
Using whichever works for you, you can run (overfitting on a batch using our interpolant)
```
[ddp stuff] main.py --overfit batch --smoke_test 0 --check_nans 0 --interpolant_type ours"
```
To load a checkpoint just add on
```
--load_model_ckpt_path ${CKPT_PATH}
```

If using CKPT the ```kth_model_ours.pt``` checkpoint use --interpolant_type ours, and if using the linear checkpoint, use --interpolant_type linear

To run a regular (non-overfitting/debugging) experiment, set --overfit batch to --overfit none

# What you should see

When monitoring samples on WANDB, check out the real_vs_generated plot. It has 4 movies: two real datapoints in the top row and two generated videos in the bottom row. For KTH, the videos are length 40, where the conditional task is to generate the next 30 after given the first 10. So the top and bottom rows should match in the first 10 frames, and the bottom row shows your generation
for the remaining 30 frames. You can change the frames per second in the plotting code in the trainer.

If training from scratch:
- overfitting on one datapoint should work in about 500-1000 steps
- batch: a few thousand 
- whole dataset: I think we ran for 250k training steps?

If loading a checkpoint, you should see samples right at initialization.


# Citation
```
@article{chen2024probabilistic,
  title={Probabilistic forecasting with stochastic interpolants and f$\backslash$" ollmer processes},
  author={Chen, Yifan and Goldstein, Mark and Hua, Mengjian and Albergo, Michael S and Boffi, Nicholas M and Vanden-Eijnden, Eric},
  journal={arXiv preprint arXiv:2403.13724},
  year={2024}
}
```

# Acknowledgements

This code builds on 

- RIVER (https://github.com/Araachie/river) (for the overall modeling approach, VQVAE checkpoints, dataloading code, and so on)
- https://github.com/willi-menapace/PlayableVideoGeneration (RIVER used this for data processing)
- https://github.com/edouardelasalles/srvp/. (RIVER used this for data processing)

