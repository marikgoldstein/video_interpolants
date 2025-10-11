

- check that you can use ffmpeg. For me on the cluster, it is "module load ffmpeg"
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
- when they run a KTH experiment, they pass in a yml config specific to their kth experiments. We need to edit that config and point it to where our path is etc . edit the kth config to find the data root path, and the pretrained autoencoder checkpoint (downloaded from the RIVER github links) 
- add an argument to train.py to get a ckpt_fname for the main model checkpoint
- over-ride the training_loop.py file in two places
- setup training args should copy ckpt_fname from args to the dictionary
- the model loading area should read ckpt_fname instead of the existing resume-step-based checkpoint names
- I Updated the load_checkpoint function in training/trainer.py and the call to load_checkpoint in training/training_loop.py to take a “manual” argument, if true use the path I specify for checkpoint directly instead of using the existing filenaming structure in the repository

- get python lib that they use for data augmentation for video data
pip install albumentations

- edit their logger to use my wandb 
In RIVER, edit lutils/logger.py to take wandb entity (marikgoldstein in my case)
If wandb logging , to log movies, pip install moviepy imageio

- training
python train.py --config configs/kth.yaml --run-name moo --ckpt_fname ckpts/river_kth_64_model.pth --wandb

You should see this at beginning of training ^ “generating frames”

Then you should see an initial eval, then training. 

By that point, the eval should log to wandb media, videos

- their evaluation code isn’t great + not all pieces are included. FVD wasn’t included. I found a separate repo to compute that. 
https://github.com/google-research/google-research/blob/20b2520e416edaea8c038bbf54cc1c739c542822/frechet_video_distance/README.md

- Parallelism: I remember for some reason i re-wrote everything with torch ddp. Just ask me to make my copy public if you need this.

