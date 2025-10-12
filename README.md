
- pip install albumentations  
- pip install moviepy imageio

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

--ckpt_fname ckpts/river_kth_64_model.pth 

- their evaluation code isn’t great + not all pieces are included. FVD wasn’t included. I found a separate repo to compute that. 
https://github.com/google-research/google-research/blob/20b2520e416edaea8c038bbf54cc1c739c542822/frechet_video_distance/README.md

In more detail, we had started off with another project's repo. That project is called RIVER (https://github.com/Araachie/river). They conditionally model frame T+1 given frame T as well as a randomly chosen frame between 1 and T-1. But, RIVER's data preprocessing code actually depended on two more older repos  https://github.com/willi-menapace/PlayableVideoGeneration and https://github.com/edouardelasalles/srvp/. Unfortunately, some of the data processing steps were not well documented in the 3 repos and some steps have become outdated. For example, the KTH black-white video dataset (https://www.csc.kth.se/cvap/actions/) seems to have recently broken (the .zip links don't work for me?) such that the data is not downloadable anymore. A subset of the data is available here https://www.kaggle.com/datasets/alexeychinyaev/kth-video-dataset but this data has some videos missing. And I don't have access to my old copy anymore because I graduated from my phd.


