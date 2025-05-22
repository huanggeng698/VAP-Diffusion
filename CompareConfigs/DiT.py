import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)
    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.train = d(
        n_steps=300000,
        batch_size=128,
        mode='cond',
        log_interval=10,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0001,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='DiT',
        num_classes=8,
        learn_sigma=False,
    )

    config.dataset = d(
        name='isic256_featuresl',
        path='/storage/ScientificPrograms/Conditional_Diffusion/U-VIT-G/assets/datasets/ISIC256_F_text_pro/',
        # path='../U-ViT-main/assets/datasets/ISIC256_features/train',
        cfg=True,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=50,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=1.0,
        path=''
    )

    return config
