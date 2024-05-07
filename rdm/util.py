import random
from functools import partial

import numpy as np
import torch
from einops import rearrange

from ldm.util import get_obj_from_str
import os
from pathlib import Path
import uuid
from PIL import Image
import torchvision
import submitit
import argparse


## Diffusion
def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")

def ischannellastimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[-1] == 3 or x.shape[-1] == 1)


def load_partial_from_config(config):
   return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def crop_coords(img_size, crop_size, random_crop: bool):
    assert crop_size <= min(img_size)
    height, width = img_size
    if random_crop:
        # random crop
        h_start = random.random()
        w_start = random.random()
        y1 = int((height - crop_size) * h_start)
        x1 = int((width - crop_size) * w_start)
    else:
        # center crop
        y1 = (height - crop_size) // 2
        x1 = (width - crop_size) // 2

    return x1, y1

def rescale(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.)/2.

def bchw_to_np(x, grid=False, clamp=False):
    if grid:
        x = torchvision.utils.make_grid(x, nrow=min(x.shape[0], 4))[None, ...]
    x = rescale(rearrange(x.detach().cpu(), "b c h w -> b h w c"))
    if clamp:
        x.clamp_(0, 1)
    return x.numpy()


def aggregate_and_noise(aggregate, r_enc, k_nn, k_nn_max_noise):
    """
    TRAIN
    if aggregate, we average the returned encodings and repeat them:
    """
    device = r_enc.device
    if torch.norm(r_enc, dim=-1).max() > 1.01:
        print("[WARN - aggregate_and_noise] \t embeddings are not normalized.", torch.norm(r_enc, dim=-1).max())
    if aggregate:
        r_enc = torch.sum(r_enc, dim=1).unsqueeze(1)
        r_enc = r_enc / k_nn
        r_enc = r_enc + torch.randn_like(r_enc, device=device) * (
            k_nn_max_noise - k_nn_max_noise * torch.rand(1, device=device)
        )
        r_enc = r_enc.repeat([1, k_nn, 1])
    else:
        r_enc = r_enc + torch.randn_like(r_enc, device=device) * (
            k_nn_max_noise - k_nn_max_noise * torch.rand(1, device=device)
        )
    return r_enc


def aggregate_and_noise_query(aggregate, r_enc, k_nn, noise_magnitude):
    """
    PREDICT
    """
    device = r_enc.device
    logging = {}
    if torch.norm(r_enc, dim=-1).max() > 1.01:
        print("[WARN - aggregate_and_noise_query] \t embeddings are not normalized.", torch.norm(r_enc, dim=-1).max())
    if aggregate:
        r_enc = torch.sum(r_enc, dim=1).unsqueeze(1)
        r_enc = r_enc / k_nn
        
        logging['agg_pre_noise_l2_norm'] = torch.norm(r_enc).cpu()
        
        r_enc = r_enc + torch.randn_like(r_enc, device=device) * (noise_magnitude)
        r_enc = r_enc.repeat([1, k_nn, 1])
    else:
        r_enc = r_enc + torch.randn_like(r_enc, device=device) * (noise_magnitude)
    return r_enc, logging

def custom_to_pil(x):
    if isinstance(x,np.ndarray):
        x = torch.from_numpy(x)
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

# used for submitit jobs
class Trainer(object):
    def __init__(self, args, main_func=None, unknown_args=None):
        self.args = args
        self.unknown_args = unknown_args
        self.main_func = main_func

    def __call__(self):
        self._setup_gpu_args()
        # code to call:
        self.main_func(self.args)

    def checkpoint(self):
        import os
        import submitit

        print("Requeuing ", self.args, self.unknown_args)
        empty_trainer = type(self)(self.args, unknown_args=self.unknown_args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")

class BatchTrainer:
    def __init__(self, main_func, args, all_jobs):
        self.args = args
        self.all_jobs = all_jobs
        self.main_func = main_func

    def __call__(self, cur_job=None):
        self._setup_gpu_args()
        self.main_func(all_jobs=self.all_jobs, args=self.args, cur_job=cur_job)

    def checkpoint(self, cur_job):
        import submitit
        print("Requeuing ", self.args, cur_job)
        empty_trainer = type(self)(self.args, self.all_jobs)
        return submitit.helpers.DelayedSubmission(empty_trainer, cur_job)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_shared_folder_aws(subfolder='shard_jobs') -> Path:
    user = os.getenv("USER")
    if Path(f"/data/home/{user}/checkpoint/").is_dir():
        p = Path(f"/data/home/{user}/checkpoint/{subfolder}")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file_aws(subfolder='shard_jobs'):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder_aws(subfolder)), exist_ok=True)
    init_file = get_shared_folder_aws() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file



def get_shared_folder(subfolder='shard_jobs') -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/{subfolder}")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(subfolder='shard_jobs'):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(subfolder)), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def submitit_experiment_runner(exp_name, main_func, parser_func):
    parser = parser_func()
    config_args = argparse.Namespace(add_help=False)
    args = parser.parse_args(namespace=config_args)

    # rank DDP stuff
    args.world_size = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    print_args(args)

    if args.job_dir == "":
        args.job_dir = get_shared_folder(exp_name) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    kwargs = {}
    if args.use_volta32:
        kwargs["slurm_constraint"] = "volta32gb"
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    executor.update_parameters(
        mem_gb=80 * args.ngpus if not args.mem_gb else args.mem_gb,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,  # one task per GPU
        cpus_per_task=10,
        nodes=args.nodes,
        timeout_min=args.timeout,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        **kwargs,
    )

    executor.update_parameters(name=exp_name)
    args.dist_url = get_init_file(exp_name).as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args, main_func=main_func)
    job = executor.submit(trainer)
    print("Submitted job_id:", job.job_id)

# Plotting + Reports

COLORS = {
    'pink':'#B43C8C',
    'purplelight': '#EDEDFF',
    'purple': '#6441D2',
    'purpledark': '#2C027D',
    'blue': '#004BB9',
    'bluemeta': '#0064E0',
    'cyan': '#0073AA',
    'teal': '#00787D',
    'green': '#007D1E',
    'gray': '#344854',
    'graydark': '#1C2B33',
    'graylight': '#CBD2D9',
    'red': '#AA0A1E',
    'orange': '#AB5710',
    'orangelight': '#F0AA19',
}
