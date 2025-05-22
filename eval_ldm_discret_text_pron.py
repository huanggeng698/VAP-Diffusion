from tools.fid_score import calculate_fid_given_paths
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
from torch.utils.data import DataLoader
import utils
from datasets import get_dataset
import tempfile
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import libs.autoencoder
import einops
import os
os.evirenment["CUDA_VISIBLE_DEVICES"]='0'

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)
    test_dataset = dataset.get_split(split='test', labeled=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True, 
                                    drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True)

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()
    
    def cfg_nnet(x, timesteps, context, label):
        _cond = nnet(x, timesteps, context=context, y=label)

        _empty_context = torch.tensor(dataset.empty_context,device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D',B=x.size(0))

        _empty_class = torch.tensor(dataset.K, device=device).unsqueeze(dim=0)
        _empty_class = einops.repeat(_empty_class, 'C -> B C',B=x.size(0)).squeeze(dim=1)
        _uncond = nnet(x,timesteps, context=_empty_context, y=_empty_class)

        return _cond + config.sample.scale *(_cond - _uncond) 

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def decode_large_batch(_batch):
        decode_mini_batch_size = 50  # use a small batch size since the decoder is large
        xs = []
        pt = 0
        for _decode_mini_batch_size in utils.amortize(_batch.size(0), decode_mini_batch_size):
            x = decode(_batch[pt: pt + _decode_mini_batch_size])
            pt += _decode_mini_batch_size
            xs.append(x)
        xs = torch.concat(xs, dim=0)
        assert xs.size(0) == _batch.size(0)
        return xs
    
    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                ### input both_text and label
                _, _context, _label = data
                yield _context, _label

    context_generator = get_context_generator()
    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    # if 'cfg' in config.sample and config.sample.cfg and config.sample.scale > 0:  # classifier free guidance
    #     logging.info(f'Use classifier free guidance with scale={config.sample.scale}')
    #     def cfg_nnet(x, timesteps, y):
    #         _cond = nnet(x, timesteps, y=y)
    #         _uncond = nnet(x, timesteps, y=torch.tensor([dataset.K] * x.size(0), device=device))
    #         return _cond + config.sample.scale * (_cond - _uncond)
    # else:
    #     def cfg_nnet(x, timesteps, y):
    #         _cond = nnet(x, timesteps, y=y)
    #         return _cond

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode={config.train.mode}, mixed_precision={config.mixed_precision}')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        noise_schedule = NoiseScheduleVP(scedule='discrete', base=torch.tenso(_betas, device=device))

        def model_fn(x, t_continuous):
            t = t_continuous * N
            eps_pre = cfg_nnet(x, t, **kwargs)
            return eps_pre

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / N, T=1.)

        return decode_large_batch(_z)

    # def sample_fn(_n_samples):
    #     if config.train.mode == 'uncond':
    #         kwargs = dict()
    #     elif config.train.mode == 'cond':
    #         kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
    #     else:
    #         raise NotImplementedError
    #     _z = sample_z(_n_samples, _sample_steps=config.sample.sample_steps, **kwargs)
    #     return decode_large_batch(_z)
    def sample_fn(_n_samples):
        def sample_fn(_n_samples):
            _context,_label = next(context_generator)
            assert _context.size(0) == _n_samples
            return dpm_solver_sample(_n_samples, config.sample.sample_steps, context=_context, label=_label=_label)

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')
        utils.sample2dir(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)
        if accelerator.is_main_process:
            fid = calculate_fid_given_paths((dataset.fid_stat, path))
            logging.info(f'nnet_path={config.nnet_path}, fid={fid}')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
