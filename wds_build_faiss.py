#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from retrieval_utils import build_faiss_index, init_index, NUM_FILES
import os
import argparse

import torch
import submitit
from rdm.util import print_args, get_shared_folder, get_init_file

import time 

JOB_NAME = 'retrieval_clip'

def get_parser():
    parser = argparse.ArgumentParser(
        description="RDM Retrieval-FAISS"
    )

    # clip model = ViT-B/32
    parser.add_argument("--clip_model", default="ViT-B-16-quickgelu", help="CLIP type", type=str)

    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=72*60, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="devlab", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument("--mem_gb", type=int, help="CPU memory per GPU")
    
    return parser



def build_index():
    index, label = init_index(use_lossless_index=False)
    pfx = str(time.time()).split(".")[0]
    build_faiss_index(index, 0, NUM_FILES, f'{pfx}_shutterstock_239M.2.{label}.faiss', train_size=1_000_000)

if __name__ == "__main__":
    parser = get_parser()
    config_args = argparse.Namespace(add_help=False)
    args = parser.parse_args(namespace=config_args)
    
    # rank DDP stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    args.world_size = n_gpus
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    print_args(args)


    if args.job_dir == "":
        args.job_dir = get_shared_folder(JOB_NAME) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    kwargs = {}
    executor.update_parameters(
        mem_gb=80 * args.ngpus if not args.mem_gb else args.mem_gb,
        gpus_per_node=args.ngpus,
        tasks_per_node=args.ngpus,  # one task per GPU
        cpus_per_task=80,
        nodes=args.nodes,
        timeout_min=args.timeout,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        **kwargs
    )
    executor.update_parameters(name=JOB_NAME)
    args.dist_url = get_init_file(JOB_NAME).as_uri()
    args.output_dir = args.job_dir


    executor.update_parameters(slurm_array_parallelism=50)


    job = executor.submit(build_index)