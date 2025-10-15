
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
