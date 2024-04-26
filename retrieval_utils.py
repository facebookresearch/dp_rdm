import os
import numpy as np
from functools import partial, cache
from multiprocessing import Pool
from tqdm import tqdm
import time

import tarfile
from torchvision import transforms
from PIL import Image
import io
import faiss
to_pil = transforms.ToPILImage()
NUM_FILES = 239_000_000
WEBDATASET_PARENT_FOLDER = '/path/to/webdataset_512/'
WEBDATASET_FOLDER_PFX = 'webdataset_folder_prefix.csv.gz'


WEBDATASET_NPZ_METACLIP_BASEPATH = '/path/to/WEBDATASET_npz'
WEBDATASET_NPZ_INDEX_PATH = '/path/to/WEBDATASET_files_index.npz'

def parse_fname(fname):
    fname_split = fname.split('/')
    return [int(i) for i in fname_split[-2].split('_')] + [int(fname_split[-1].split('.')[0])]

def load_shard_part_npz(shard_no, part_no, subpart_no, file_no, basepath):
    path = f'{basepath}/{shard_no}_{part_no}_{subpart_no}/{file_no:06}.npz'
    return np.load(path)['arr_0']

def load_embedding_from_idx(idx, npz_index_path=WEBDATASET_NPZ_INDEX_PATH):
    file_index = load_file_index(npz_index_path)
    parts = file_index[idx]
    return process_part(parts[0:4])[parts[-1]]
    
def load_embeddings_mp(start_idx, end_idx, npz_index_path=WEBDATASET_NPZ_INDEX_PATH, basepath=WEBDATASET_NPZ_METACLIP_BASEPATH, n_processes=80):
    """
    Loads embeddings in parallel from a specified index range using multiprocessing.

    Args:
    - start_idx (int): Starting index of the range.
    - end_idx (int): Ending index of the range.
    - basepath (str): Base path for loading embeddings.

    Returns:
    - numpy.ndarray: Concatenated embeddings within the specified index range.

    Note:
    Utilizes multiprocessing to concurrently load embeddings from the specified index range
    using `parts_from_idx_range(start_idx, end_idx)` to define partitions.
    """
    parts = parts_from_idx_range(start_idx, end_idx, npz_index_path)
    
    # omit the actual files here
    parts_fetched = np.unique(parts[:,0:4], axis=0)
    print(parts_fetched)
    pool = Pool(processes=n_processes)  # Create a pool of processes
    results = pool.map(partial(process_part, basepath=basepath), parts_fetched)  # Map the function to different parts using the pool
    pool.close()
    pool.join()
    
    embeddings =  np.concatenate(results, axis=0)
    n_embeddings = end_idx - start_idx
    # start at the right point and end accordingly
    return embeddings[parts[0,-1]:parts[0,-1] + n_embeddings]

@cache
def load_file_index(index_path):
    return np.load(index_path)['arr_0']

def process_part(part_info, basepath=WEBDATASET_NPZ_METACLIP_BASEPATH):
    embeddings = load_shard_part_npz(*part_info, basepath)
    return embeddings

def parts_from_idx_range(start_idx, end_idx, index_path='files_index.npz'):
    index = load_file_index(index_path)  
    parts = index[start_idx:end_idx]
    return parts

def fetch_and_normalize_embeddings(start, end, npz_index_path=WEBDATASET_NPZ_INDEX_PATH, basepath=WEBDATASET_NPZ_METACLIP_BASEPATH, n_processes=80, **kwargs):
    print(start, end )
    embeddings = load_embeddings_mp(start, end, npz_index_path=npz_index_path, basepath=basepath, n_processes=n_processes).astype('float32')
    indices = np.arange(start, start + embeddings.shape[0])
    # normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    return embeddings, indices
    
def build_faiss_index(index, rec_start, rec_end, faiss_index_path, num_keys_to_add_at_a_time=1000_000, train_size=0, **kwargs):
    start_time = time.time()
    
    if train_size == 0:
        train_size = rec_end - rec_start

    print("Training index...")
    to_add, _ = fetch_and_normalize_embeddings(0, train_size, **kwargs)
    index.train(to_add)
    faiss.write_index(index, faiss_index_path)
    
    dstore_size = rec_end - rec_start

    batch_size = min(num_keys_to_add_at_a_time, dstore_size)
    n_batches = dstore_size // batch_size
    start = rec_start
    
    for batch in range(n_batches):
        end = min(rec_end, start + batch_size)
        print(f'[{batch}] - Fetching {start} - {end}')

        to_add, indices = fetch_and_normalize_embeddings(start, end, **kwargs)
        index.add_with_ids(to_add.astype(np.float32), indices)
        print(f'Added {index.ntotal:,} samples so far', f'\t {(time.time() - start_time):03f}s')
        start = end - 1
        
        faiss.write_index(index, faiss_index_path)
    
    print("Saving index.. \t", faiss_index_path, f'\t {(time.time() - start_time):03f}s')
    faiss.write_index(index, faiss_index_path)


def init_index(
    use_lossless_index = True,
    embedding_dimension = 512,
    code_size = 256,
    nbits = 8,
    ncentroids = 8192,
    probe = 256):


    if use_lossless_index:
        index = faiss.index_factory(embedding_dimension, 'IDMap,Flat', faiss.METRIC_INNER_PRODUCT)
        faiss_label='lossless'
    else:
        quantizer = faiss.IndexFlatL2(embedding_dimension)
        index = faiss.IndexIVFPQ(quantizer, embedding_dimension, ncentroids, code_size, nbits)
        index.nprobe = probe
        faiss_label = f'IVFPQ_cs{code_size}_probe{probe}_nc{ncentroids}'
    
    return index, faiss_label


def webdataset_tar_path(shard_parts):
    path = f'{WEBDATASET_PARENT_FOLDER}{WEBDATASET_FOLDER_PFX}_{shard_parts[0]}_{shard_parts[1]}_{shard_parts[2]}.csv/{shard_parts[3]:06}.tar'
    return path

def webdataset_image_from_parts(shard_parts):
    url_base = webdataset_tar_path(shard_parts)
    tar = tarfile.open(url_base)
    count = 0
    file_no = shard_parts[-1]
    for member in tar.getmembers():
        if 'jpg' in member.name:
            if file_no == count:
                print(url_base, member)
                f=tar.extractfile(member)
                img = Image.open(io.BytesIO(f.read()))
                return img
            count += 1
