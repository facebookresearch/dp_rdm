import numpy as np
import webdataset as wds
from pathlib import Path
import os
import argparse

from rdm.modules.retrievers import ClipImageRetriever 
from torchvision import transforms
import torch
import submitit
from functools import partial
from rdm.util import print_args, get_shared_folder_aws, get_init_file_aws

from tqdm import tqdm

preproc = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()])

JOB_NAME = 'shutter_clip'
WDS_BATCH_SIZE = 4000
PARENT_FOLDER = '/fsx-shutterstock-image/dataset/first_cleaned/ss-photo-bucket/webdataset_512/'
FOLDER_PFX = 'meta_sstk_non_vector_non_editorial_non_model_release_non_mature_images_metadata.csv.gz'
SHUTTERSTOCK_WDS_PATH = PARENT_FOLDER + FOLDER_PFX + '_{0..7}_{0..7}_{0..1}.csv/{000000..000376}.tar'

def get_parser():
    parser = argparse.ArgumentParser(
        description="RDM Shutterstock-CLIP"
    )
    # npz path
    parser.add_argument("--npz_path", default='/data/home/lebensold/checkpoint/shutterstock_npz', type=str, help="")
    parser.add_argument("--clip_pretrained", default="models/metaclip/b16_400m.pt", help="CLIP model to load", type=str)
    # clip_type = clip
    parser.add_argument("--clip_type", default="open_clip", help="CLIP type", type=str)

    # clip model = ViT-B/32
    parser.add_argument("--clip_model", default="ViT-B-16-quickgelu", help="CLIP type", type=str)

    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=72*60, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")
    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument("--mem_gb", type=int, help="CPU memory per GPU")
    
    return parser



def init_dataset_loader(url_base):
    dataset = wds.DataPipeline(
        wds.SimpleShardList(url_base),
        # at this point we have an iterator over all the shards
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker
        wds.tarfile_to_samples(),
        wds.decode("pilrgb"),
        # at this point, we have an list of decompressed training samples from each shard in this worker in sequence
        wds.to_tuple("jpg", "__url__", "__key__"),
        wds.map_tuple(preproc,),
        wds.batched(WDS_BATCH_SIZE)
    )
    return wds.WebLoader(dataset)



def shard_tar_to_npz(model, npz_path, shard_parts, tar_no):
    url_base = f'{PARENT_FOLDER}{FOLDER_PFX}_{shard_parts[0]}_{shard_parts[1]}_{shard_parts[2]}.csv/{tar_no:06}.tar'
    save_base_path = f'{npz_path}/{shard_parts[0]}_{shard_parts[1]}_{shard_parts[2]}'
    save_path = f'{save_base_path}/{tar_no:06}.npz'

    p = Path(save_base_path)
    p.mkdir(exist_ok=True)
    
    if os.path.isfile(save_path):
        print(f"[{shard_parts} - {tar_no:06}] \t - Skipping \t {save_path} found")
        return

    if not os.path.isfile(url_base):
        print(f"[{shard_parts} - {tar_no:06}] \t - Skipping \t {url_base} not found")
        return
        
    loader = init_dataset_loader(url_base)
    data_iterator = iter(loader)
    embeddings = []
    with torch.no_grad():
        for data in data_iterator:
            data_batch = data[0][0].to(model.device)
            image_features = model(data_batch)
            embeddings.append(image_features.cpu())
            print(f'[{shard_parts} - {tar_no:06}] \t Adding', len(image_features))
    final_embeddings = torch.cat(embeddings, dim=0)
    print(f'[{shard_parts} - {tar_no:06}] - {final_embeddings.shape}')
    np.savez_compressed(save_path, final_embeddings)


def shard_to_npz_batch(cur_chunk, args, all_chunks):
    model = ClipImageRetriever(model=args.clip_model, clip_type=args.clip_type, pretrained=args.clip_pretrained, device='cuda')    
    chunks = all_chunks[cur_chunk]
    for part in chunks:
        shard_tar_to_npz(model, npz_path=args.npz_path, shard_parts=(part[0], part[1], part[2]), tar_no=part[3])

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
        args.job_dir = get_shared_folder_aws(JOB_NAME) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    kwargs = {}
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
        **kwargs
    )
    executor.update_parameters(name=JOB_NAME)
    args.dist_url = get_init_file_aws(JOB_NAME).as_uri()
    args.output_dir = args.job_dir


    executor.update_parameters(slurm_array_parallelism=50)

    shard_tar_parts = []
    for idx in range(8):
        for jdx in range(8):
            for kdx in range(2):
                for ldx in range(377):
                    shard_tar_parts.append((idx, jdx, kdx, ldx))
    

    chunk_size = 20
    start = 0
    chunks = []
    while start < len(shard_tar_parts):
        chunks.append(shard_tar_parts[start:start+chunk_size])
        start = start + chunk_size
    
    jobs = executor.map_array(partial(shard_to_npz_batch, args=args, all_chunks=chunks), np.arange(len(chunks)))