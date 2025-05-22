## VAP-Diffusion：Enriching Descriptions with MLLMs for Enhanced Medical Image Generation



framework introduction:...
--------------------



This codebase implements ...(summary)


💡This codebase contains:
* An implementation of [VAP-Diffusion](libs/uvit_t2i_label.py) with optimized attention computation
* Efficient training scripts for [pixel-space diffusion models](train.py), [latent space diffusion models](train_ldm_discrete.py) and [text-to-image diffusion models](train_t2i_discrete.py)
* Efficient evaluation scripts for [pixel-space diffusion models](eval.py) and [latent space diffusion models](eval_ldm_discrete.py) and [text-to-image diffusion models](eval_t2i_discrete.py)


## Dependency

```sh
pip install -r requirements.txt
```

## Pretrained Models


|                                                         Model                                                          |  FID  | training iterations | batch size |
|:----------------------------------------------------------------------------------------------------------------------:|:-----:|:-------------------:|:----------:|
|      [ChestXray-14 (VAP-Diff)](about:blank)      | xxxx  |        xxxK         |    xxx     |
|      [ISIC2018 (VAP-Diff)](about:blank)      | xxxx  |        xxxK         |    xxx     |
|      [Colonoscopy (VAP-Diff)](about:blank)      | xxxx  |        xxxK         |    xxx     |





## Preparation Before Training and Evaluation

#### Autoencoder
Download `stable-diffusion` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains image autoencoders converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion)). 
Put the downloaded directory as `assets/stable-diffusion` in this codebase.
The autoencoders are used in latent diffusion models.

#### Data
Prepare the dataset, including the chest X-ray dataset [Chest-Xray 14](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf), and the dermatology datasets [ISIC2018](https://challenge.isic-archive.com/data/#2018) and [ISIC2019](https://challenge.isic-archive.com/data/#2019).The colonoscopy dataset is a private dataset involving certain privacy concerns and therefore cannot be made publicly available.

Then extract their features according to `scripts/extract_mscoco_feature.py` `scripts/extract_test_prompt_feature.py` `scripts/extract_empty_feature.py`.

#### Reference statistics for FID
Download `fid_stats` directory from this [link](https://drive.google.com/drive/folders/1CNwcf5u_5WvnWy3qX_WLcY_F5BLViM_6?usp=drive_link) (which contains reference statistics for FID).
Put the downloaded directory as `assets/fid_stats` in this codebase.
In addition to evaluation, these reference statistics are used to monitor FID during the training process.

## Training
We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library to help train with distributed data parallel and mixed precision. The following is the training command:
```sh
# the training setting
num_processes=2  # the number of gpus you have, e.g., 2
train_script=train_t2i_discrete.py  # the train script, one of <train.py|train_ldm.py|train_ldm_discrete.py|train_t2i_discrete.py>
                       # train.py: training on pixel space
                       # train_ldm.py: training on latent space with continuous timesteps
                       # train_ldm_discrete.py: training on latent space with discrete timesteps
                       # train_t2i_discrete.py: text-to-image training on latent space
config=configs/COT_ISIC256_uvit_text_label.py  # the training configuration
                                      # you can change other hyperparameters by modifying the configuration file

# launch training
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 $train_script --config=$config
```


We provide all commands to reproduce U-ViT training in the paper:
```sh
# CIFAR10 (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/cifar10_uvit_small.py

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/celeba64_uvit_small.py 

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_mid.py

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_large.py

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet256_uvit_large.py

# ImageNet 256x256 (U-ViT-H/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet256_uvit_huge.py

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet512_uvit_large.py

# ImageNet 512x512 (U-ViT-H/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet512_uvit_huge.py

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16
```



## Evaluation (Compute FID)






We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library for efficient inference with mixed precision and multiple gpus. The following is the evaluation command:
```sh
# the evaluation setting
num_processes=2  # the number of gpus you have, e.g., 2
eval_script=eval.py  # the evaluation script, one of <eval.py|eval_ldm.py|eval_ldm_discrete.py|eval_t2i_discrete.py>
                     # eval.py: for models trained with train.py (i.e., pixel space models)
                     # eval_ldm.py: for models trained with train_ldm.py (i.e., latent space models with continuous timesteps)
                     # eval_ldm_discrete.py: for models trained with train_ldm_discrete.py (i.e., latent space models with discrete timesteps)
                     # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
config=configs/cifar10_uvit_small.py  # the training configuration

# launch evaluation
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 eval_script --config=$config
```
The generated images are stored in a temperary directory, and will be deleted after evaluation. If you want to keep these images, set `--config.sample.path=/save/dir`.


We provide all commands to reproduce FID results in the paper:
```sh
# CIFAR10 (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_small.py --nnet_path=cifar10_uvit_small.pth

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/celeba64_uvit_small.py --nnet_path=celeba64_uvit_small.pth

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_mid.py --nnet_path=imagenet64_uvit_mid.pth

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_large.py --nnet_path=imagenet64_uvit_large.pth

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet256_uvit_large.py --nnet_path=imagenet256_uvit_large.pth

# ImageNet 256x256 (U-ViT-H/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm_discrete.py --config=configs/imagenet256_uvit_huge.py --nnet_path=imagenet256_uvit_huge.pth

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet512_uvit_large.py --nnet_path=imagenet512_uvit_large.pth

# ImageNet 512x512 (U-ViT-H/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm_discrete.py --config=configs/imagenet512_uvit_huge.py --nnet_path=imagenet512_uvit_huge.pth

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=mscoco_uvit_small.pth

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16 --nnet_path=mscoco_uvit_small_deep.pth
```




## References
If you find the code useful for your research, please consider citing
```bib
@inproceedings{bao2022all,
  title={VAP-Diffusion：Enriching Descriptions with MLLMs for Enhanced Medical Image Generation},
  author={Anonymized},
  booktitle = {MICCAI},
  year={2025}
}
```

This implementation is based on
* [U-Vit](https://github.com/baofff/U-ViT) 
