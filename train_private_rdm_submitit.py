# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

from train_private_rdm import get_parser
from rdm.util import get_shared_folder, get_init_file
import submitit


def parse_args():
    parser = get_parser()
    parser = argparse.ArgumentParser("Submitit for private RDM", parents=[parser], add_help=False)
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=8, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=72*60, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument("--mem_gb", type=int, help="CPU memory per GPU")
    return parser.parse_known_args()


class Trainer(object):
    def __init__(self, args, unknown_args=None):
        self.args = args
        self.unknown_args = unknown_args

    def __call__(self):
        import train_private_rdm

        self._setup_gpu_args()
        self._setup_lightning_args()
        train_private_rdm.main(self.args, self.unknown_args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args, self.unknown_args)
        empty_trainer = type(self)(self.args, unknown_args=self.unknown_args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
    
    # added for lightning
    def _setup_lightning_args(self):
        import glob

        self.args.num_nodes = self.args.nodes
        self.args.gpus = ",".join([str(n) for n in range(self.args.ngpus)])
        self.args.logdir = str(self.args.output_dir)

        if self.args.output_dir.exists() and self.args.resume == "":
            try:
                self.args.resume = glob.glob(f"{self.args.logdir}/*/checkpoints/last.ckpt")[0]
                self.args.name = ""
            except IndexError:
                print("WARNING: last.ckpt not present, starting from scratch")
                self.args.resume = ""


def main():
    args, unknown_args = parse_args()

    if args.job_dir == "":
        args.job_dir = get_shared_folder('rdm') / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=80 * num_gpus_per_node if not args.mem_gb else args.mem_gb,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="train_rdm")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    print(args, unknown_args)
    trainer = Trainer(args, unknown_args=unknown_args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()