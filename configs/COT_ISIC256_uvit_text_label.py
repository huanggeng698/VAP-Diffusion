import ml_collections

###在这个版本中我们在原始UVIT文生图框架的基础上引入了label embedding
###将label embedding转为label prototype
def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        # scale_factor=0.23010
    )

    config.train = d(
        n_steps=500000,
        batch_size=256,
        log_interval=10,
        # eval_interval=5000,
        # save_interval=50000,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_t2i_label_pron',
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=512,
        depth=16,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        clip_dim=768,
        num_clip_token=256,
        num_classes=8 # num_classes = num_classes + 1
    )

    config.dataset = d(
        name='isic256_featurestextl',
        path='ScientificPrograms/Conditional_Diffusion/U-VIT-G/assets/datasets/ISIC256_F_text_pro',
        cfg=True,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=50,
        cfg=True,
        scale=1.0, 
        path=''
    )

    return config
