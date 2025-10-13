

class Config:

    def __init__(self, dataset, overfit, smoke_test, check_nans, interpolant_type, load_model_ckpt_path, wandb_entity, wandb_project):

        
        self.dataset = dataset
        self.overfit = overfit
        self.smoke_test = smoke_test
        self.interpolant_type = interpolant_type
        self.load_model_ckpt_path = load_model_ckpt_path
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.check_nans = check_nans

        REAL = (not overfit) and (not smoke_test)

        if self.dataset == 'kth':
            self.data_path =  "/mnt/home/mgoldstein/ceph/video/data/kth/hdf5s"
        elif self.dataset == 'clevrer':
            assert False
        else:
            assert False

        self.results_dir = "/mnt/home/mgoldstein/ceph/video/results/kth"


        self.epochs = 1000000000  # use num_training_steps instead

        if smoke_test:
            self.limit_train_batches = 3
            self.num_training_steps = 100
        else:
            self.limit_train_batches = -1 # per epoch. <0 means not in use. Useful for debugging epoch loops.
            self.num_training_steps = 400_000

        self.global_batch_size = 256 if REAL else 4
        self.num_workers = 4
        self.global_seed = 0
        self.update_ema_after = 10_000 # dont average bad early models into EMA weights
        self.update_ema_every = 1 
        
        # logging, ckpt'ing, sampling

        if smoke_test:
            self.print_every = 10
            self.log_every = 10
            self.save_every = 50
            self.save_most_recent_every = 50
            self.sample_every = 50
            self.num_sampling_steps = 10
        else:
            self.print_every = 100 # to terminal
            self.log_every = 100 # to wandb        
            self.save_every = 25_000
            self.save_most_recent_every = 1000
            self.sample_every = 5000 if REAL else 1000
            self.num_sampling_steps = 100

        self.time_min_sample = 1e-4
        self.time_max_sample = 1 - 1e-4
        self.time_min_training = 1e-4
        self.time_max_training = 1 - 1e-4

        self.base_lr = 1e-4
        self.min_lr = 1e-6
        self.lr_warmup_steps = 10_000 if REAL else 0
        self.lr_schedule = 'constant' # with warmup
        self.grad_clip_norm = 1.0
        self.weight_decay = 0.0

        # lr_sched_gamma: .99
        # lr_sched_step_size: 10000
        # num_training_steps: 3000000

        if self.dataset == 'kth':
            self.C_data = 3
            self.H_data = 64
            self.W_data = 64
            self.C_latent = 4
            self.H_latent = 8
            self.W_latent = 8
            self.input_size = 64
            self.crop_size = 64
            self.num_observations = 40
            self.skip_frames = 0
            self.condition_frames = 10
            self.frames_to_generate = 30
            # augmentation
            self.data_aug = False 
            self.data_random_horizontal_flip = False
            self.data_albumentations = True
            # VQVAE (in this case, the one from Taming Transformers, has its own config)
            self.vqvae_type = "ldm-vq" 
            self.vqvae_config = "f8_small"
            self.load_vqvae_ckpt_path = '/mnt/home/mgoldstein/ceph/video/ckpts/kth_vqvae.ckpt'


            # MAIN MODEL
            self.model_state_size = 4
            self.model_state_res = [8,8]
            self.model_inner_dim = 768
            self.model_depth = 4
            self.model_mid_depth = 5
            self.model_out_norm = 'ln'


        elif self.dataset == 'clevrer':
            self.C_data = 3
            self.H_data = 128
            self.W_data = 128
            self.C_latent = 4
            self.H_latent = 8
            self.W_latent = 8
            self.input_size = 64
            self.crop_size = 64
            self.num_observations = 16
            self.skip_frames = 4
            self.condition_frames = 2
            self.frames_to_generate = 14
            
            # augmentation
            self.data_aug = True
            self.data_random_horizontal_flip = False
            self.data_albumentations = False

            # VQVAE (in this case, the one from RIVER paper, config specified here)
            self.vqvae_type = "river" 
            self.vqvae_river_encoder_in_channels = 3
            self.vqvae_river_encoder_out_channels = 4
            self.vqvae_river_vector_quantizer_num_embeddings = 8192
            self.vqvae_river_vector_quantizer_embedding_dimension = 4
            self.vqvae_river_decoder_in_channels = 4
            self.vqvae_river_decoder_out_channels = 3
            self.load_vqvae_ckpt_path = '/mnt/home/mgoldstein/ceph/video/ckpts/clevrer_vqvae.ckpt'

            # MAIN MODEL
            self.model_state_size = 4
            self.model_state_res = [16,16]
            self.model_inner_dim = 768
            self.model_depth = 4
            self.model_mid_depth = 5
            self.model_out_norm = 'ln'

        else:
            assert False

