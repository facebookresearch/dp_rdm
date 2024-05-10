#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import datetime
import os
import sys
import time
from argparse import ArgumentParser
from glob import glob
from multiprocessing import cpu_count

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from ldm.util import instantiate_from_config, parallel_data_prefetch

from rdm.data.base import PatcherDataset

SEARCHERS_BASE = "searchers"

class RestartSampler(Sampler):
    def __init__(self, data_source, start_id):
        self.data_source = data_source
        self.start = start_id

    def __iter__(self):
        return iter(range(self.start,len(self.data_source)))

    def __len__(self):
        return len(self.data_source)-self.start


def load_data(config, split='train'):
    data = instantiate_from_config(config)
    if config.target == 'main.DataModuleFromConfig':
        data.prepare_data()
        data.setup()

        data = data.datasets[split]

    return data

def subsample_index(dset_size=438_000_000, p=1/1_000, device='cuda:0'):
    """
    Subsamples indices from a dataset based on a binomial distribution.

    Args:
    - dset_size (int): Size of the dataset.
    - p (float): Probability of selecting an index based on a binomial distribution.

    Returns:
    - selection (numpy.ndarray): Array of indices selected based on the binomial distribution.
    """
    selection = torch.where(torch.rand(dset_size, device=device) < p)[0].cpu().numpy()
    return selection

class DatasetBuilder(object):
    def __init__(self,
                 retriever_config,
                 data,
                 metric='dot_product',
                 patch_size=128,
                 n_patches=None,
                 batch_size=10,
                 patch_sampling='random',
                 k=10,
                 img_size=None,
                 num_workers=None,
                 max_pool_size=None,
                 visualize=False,
                 save=True,
                 saved_embeddings=None,
                 trainset_size_partitioning=None,
                 chunk_size=None,
                 gpu=True,
                 load_patch_dataset=True,
                 patch_dset_kwargs=None,
                 searcher_savepath=None,
                 timestamp_searcher_savepath=False,
                 savepath_postfix=None,
                 save_searcher=False,
                 db_engine='faiss',
                 subsample_rate=1.
                 ):
        self.retriever_config = retriever_config
        self.retriever_name = retriever_config.target.split('.')[-1]
        self.visualize = visualize
        self.distance_metric = metric
        self.db_engine = db_engine
        self.subsample_rate=subsample_rate
        # if not trainset_size_partitioning:
        #     trainset_size_partitioning = max_pool_size // 10
        # self.partitioning_trainsize = trainset_size_partitioning
        self.k = k
        self.chunk_size = chunk_size
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.load_patch_dataset = load_patch_dataset
        self.max_pool_size = max_pool_size
        self.patch_size = patch_size
        self.save_searcher = save_searcher

        # get dataset
        if self.load_patch_dataset:
            self.dset = load_data(data)
            self.dset_name = self.dset.__class__.__name__
            if n_patches is None:
                # all images in dataset are used
                if patch_sampling == "annotation":
                    # this assumes the same interface as OpenImagesBBoxBase
                    image_ids = [self.dset.get_image_id(n) for n in range(len(self.dset))]
                    n_patches = [len(self.dset.get_annotations(id)) for id in image_ids]
                    if max_pool_size is None:
                        self.max_pool_size = np.sum(n_patches)
                        print(f"Setting max pool size to {self.max_pool_size:,}")
                elif self.max_pool_size is None:
                    print('Using exactly one patch per image as max_pool_size will be exactly the size of the dataset')
                    n_patches = [1 for _ in range(len(self.dset))]
                else:
                    n_patches = [a.shape[0] for a in np.array_split(np.arange(self.max_pool_size),len(self.dset))]


            if patch_dset_kwargs is None:
                patch_dset_kwargs = dict()

            self.patch_dset = PatcherDataset(self.dset, patch_size=patch_size,
                                             n_patches=n_patches, sampling_method=patch_sampling,
                                             img_size=img_size, **patch_dset_kwargs)
            if self.max_pool_size is None:
                self.max_pool_size = len(self.patch_dset)


            patch_div = n_patches if isinstance(n_patches, int) else max(n_patches)
            self.dset_batch_size = batch_size // patch_div
            print(f'Batch size: {batch_size}; patch_div: {patch_div}; dset_batch_size: {self.dset_batch_size}')
            if num_workers is None:
                num_workers = 2 * self.dset_batch_size
            self.num_workers = num_workers
        else:
            print(f'WARNING: not loading patch_dset in {self.__class__.__name__}')

            self.dset = None
            self.patch_dset = None
            self.dset_batch_size = None
            self.num_workers = None
            self.dset_name = data.target.split('.')[-1]

        self.retriever_bs = batch_size

        # for saving of files

        assert self.max_pool_size is not None, 'Max pool size still None --> check implementation'
        # load the retriever model, which will transform the inputs to the space where the metric is calculated
        gpu = gpu and torch.cuda.is_available()
        self.retriever = self.load_retriever(gpu=gpu)
        self.gpu = gpu

        self.save_embeddings = save
        self.data_pool = {'embedding': [],
                          'img_id': [],
                          'patch_coords': []}
        self.saved_embeddings = saved_embeddings

        if self.chunk_size is not None:
            assert self.chunk_size % batch_size == 0, '"batch_size" has to evenly divide "chunk_size", if the latter is specified'
            assert self.chunk_size < self.max_pool_size

        if self.saved_embeddings:
            self.load_embeddings()

        self.searcher = None
        self.savepath_postfix = savepath_postfix
        self.searcher_savedir = searcher_savepath
        if self.searcher_savedir is None:
            searcher_savedir = data.target.split('.')[-1]+f'_{int(self.max_pool_size*1e-6):04d}M_{self.patch_size}'
            if timestamp_searcher_savepath:
                searcher_savedir = f"{self.timestamp}_{searcher_savedir}"
            if savepath_postfix is not None:
                searcher_savedir = f"{searcher_savedir}_{savepath_postfix}"
            self.searcher_savedir = os.path.join(SEARCHERS_BASE, searcher_savedir)
            print(f'No predefined savedir for searcher, setting to {self.searcher_savedir}')

        self.dir_identifier = '-'.join([self.timestamp, self.dset_name, self.retriever_name,str(patch_size)])
        if savepath_postfix is not None:
            self.dir_identifier += f"-{savepath_postfix}"

        # self.additional_embedders = {}
        # if additional_embedders:
        #     for key in tqdm(additional_embedders,desc='Loading additional embedders'):
        #         print(f'Loading embedder with key {key} and target {additional_embedders[key].target}')
        #         self.additional_embedders[key] = instantiate_from_config(additional_embedders[key])

    def load_single_file(self,saved_embeddings):
        assert saved_embeddings.endswith('.npz'), 'saved embeddings not stored as a .npz file'
        compressed = np.load(saved_embeddings)
        self.data_pool = {key: compressed[key] for key in compressed.files}
        if self.data_pool['embedding'].shape[0] >= self.max_pool_size:
            self.max_pool_size = self.data_pool['embedding'].shape[0]
        print('Finished loading of patch embeddings.')

    def load_multi_files(self,data_archive):
        out_data = {key: [] for key in self.data_pool}
        for d in tqdm(data_archive,desc=f'Loading datapool from {len(data_archive)} individual files.'):
            for key in d.files:
                out_data[key].append(d[key])

        return out_data



    def load_embeddings(self):
        if len(self.data_pool['embedding']) > 0:
            return

        print(f'Load saved patch embedding from "{self.saved_embeddings}"')



        # set timestamp to the appropriate value
        self.timestamp = '-'.join(self.saved_embeddings.rstrip('/').split('/')[-1].split('-')[:5])
        print(f'Setting timestamp to "{self.timestamp}"')

        # if not os.path.isdir(self.saved_embeddings) and 'compvis-nfs/user/ablattma/projects/sbgm/ldm/logs/retrieval_datasets' in self.saved_embeddings:
        #     print('*' * 50, "WARNING", '*' * 50)
        #     print(f'Directory {self.saved_embeddings} has been moved, remapping base path')
        #     print('*' * 110)
        #     reldirname = self.saved_embeddings.split('compvis-nfs/user/ablattma/projects/sbgm/ldm/logs/retrieval_datasets/')[-1].strip('/')
        #     self.saved_embeddings = '/export/compvis-nfs/group/datasets/retrieval_datasets/' + reldirname
        #     print(f'remapped path is {self.saved_embeddings}')
        #     if not os.path.isdir(self.saved_embeddings):
        #         raise ValueError(f'no database found under remapped path {self.saved_embeddings}. Please check or change config manually')

        if os.path.isfile(self.saved_embeddings):
            self.load_single_file(self.saved_embeddings)
        elif os.path.isdir(self.saved_embeddings):
            files = glob(os.path.join(self.saved_embeddings,'*.npz'))
            if len(files) == 1:
                self.load_single_file(files[0])
            else:
                data = [np.load(f) for f in files]
                prefetched_data = parallel_data_prefetch(self.load_multi_files,data,
                                                         n_proc=min(len(data),cpu_count()),target_data_type='dict')

                self.data_pool = {key: np.concatenate([od[key] for od in prefetched_data],axis=1)[0] for key in self.data_pool}
        else:
            raise ValueError(f'Embeddings string "{self.saved_embeddings}" nor directory neither file --> check this.')

        print(f'Finished loading of retrieval database of length {self.data_pool["embedding"].shape[0]}.')

    def save_datapool(self,postfix:str=None):
        print(f'Save embeddings...')
        shape = list(self.data_pool['embedding'][0].shape)
        shape[0] *= len(self.data_pool['embedding'])
        identifier = 'x'.join([str(s) for s in shape])
        # identifier = '-'.join([self.timestamp, self.dset_name, self.retriever_name, embedding_shape])

        if postfix:
            print(f'Adding postfix "{postfix}" to identifier')
            identifier=identifier+'-'+postfix + self.db_engine
        img_dir = f'export/datasets/retrieval_datasets/{self.dir_identifier}'
        os.makedirs(img_dir, exist_ok=True)
        self.saved_embeddings = img_dir
        saved_embeddings = f'{img_dir}/{identifier}.npz'
        self.data_pool = {key: np.concatenate(self.data_pool[key]) for key in self.data_pool}

        np.savez_compressed(saved_embeddings, **self.data_pool)
                            # embedding=self.data_pool['embedding'],
                            # img_id=self.data_pool['img_id'],
                            # patch_coords=self.data_pool['patch_coords']
                            # )
        return saved_embeddings

    def reset_data_pool(self):
        self.data_pool = {key : [] for key in self.data_pool}

    def custom_collate(self):
        def custom_coll(elems):
            return torch.cat(elems,0)

        if isinstance(self.patch_dset.n_patches,(list,tuple,np.ndarray)):
            def collate_fn(batch):
                elem = batch[0]
                elem_type = type(elem)
                collated = dict()
                for key in elem:
                    if key in ["patch", "patch_coords", "img_id"]:
                        collated[key] = custom_coll([
                            torch.as_tensor(d[key]) for d in batch
                        ])
                    else:
                        try:
                            collated[key] = default_collate([
                                d[key] for d in batch
                            ])
                        except RuntimeError:
                            # images may have differing sizes
                            collated[key] = [d[key] for d in batch]

                try:
                    # elem_type should be dict, there may
                    # be some reason to have this here
                    # keeping it for legacy
                    collated = elem_type(collated)
                except TypeError:
                    # The mapping type may not support `__init__(iterable)`.
                    # return dict in that case
                    pass

                return collated

                # try:
                #     return elem_type({
                #         key: custom_coll([torch.as_tensor(d[key]) for d in batch])
                #              if key in ['patch', 'patch_coords', 'img_id'] else
                #              default_collate([d[key] for d in batch])
                #         for key in elem})
                # except TypeError:
                #     # The mapping type may not support `__init__(iterable)`.
                #     return {
                #         key: custom_coll([d[key] for d in batch])
                #              if key in ['patch', 'patch_coords', 'img_id'] else
                #              default_collate([d[key] for d in batch])
                #         for key in elem}

            return collate_fn
        else:
            return None

    def build_data_pool(self):
        '''TODO: This method can be significantly simplified'''
        start_ex = 0
        n_examples = 0
        start_loader_it = 0
        entire_dset = isinstance(self.patch_dset.n_patches, (list, tuple))
        if self.saved_embeddings:
            current_len = self.data_pool['embedding'].shape[0]
            if current_len >= self.max_pool_size:
                print('embeddings are already saved, not recomputing....')
                return
            else:
                print(f'Restarting extraction as only {current_len} of overall {self.max_pool_size} examples are in data_pool.')
                n_examples = current_len
                if entire_dset:
                    sample_counter = np.cumsum(self.patch_dset.n_patches)
                    loader_counter = sample_counter[self.dset_batch_size-1::self.dset_batch_size]
                    start_loader_it = int(np.argwhere(loader_counter>n_examples)[0])
                    start_ex = int(np.argwhere(sample_counter>n_examples)[0])
                else:
                    start_ex = current_len // self.retriever_bs + 1
                    start_loader_it = start_ex
                self.data_pool = {key: [] for key in self.data_pool}

        sampler=RestartSampler(self.patch_dset,start_ex)
        loader = DataLoader(self.patch_dset,
                            self.dset_batch_size,
                            sampler=sampler,
                            num_workers=self.num_workers,
                            collate_fn=self.custom_collate(),
                            drop_last=False if entire_dset else True)
        print(f'starting extraction of feature embeddings from iteration {start_loader_it} of dataloader')
        part = int(n_examples/self.chunk_size) + 1 if self.chunk_size is not None else 1
        total = len(loader) if entire_dset else (self.max_pool_size-n_examples) // self.retriever_bs
        deltas = []
        deltas_per_sec = []
        overall_start = time.time()
        try:
            for batch in (pbar := tqdm(loader, desc='Extracting feature embeddings',
                                       total=total)):
                if 'patch' in batch:
                    patches = batch['patch']
                else:
                    if self.save_embeddings:
                        self.save_datapool(postfix=f'part_{part}')
                        self.reset_data_pool()
                    break
                # img_ids = rearrange(batch['img_id'], 'b n -> (b n)').numpy()
                # patch_coords = rearrange(batch['patch_coords'], 'b n k -> (b n) k').numpy()
                img_ids = batch['img_id'].numpy()
                patch_coords = batch['patch_coords'].numpy()

                embeddings = self.embed(patches)
                start = time.time()
                if len(embeddings.shape) == 3:
                    print('reshaping', embeddings.shape)
                    embeddings = embeddings.reshape([embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2]])
        
                
                delta = (time.time() - start)
                deltas.append(delta)
                deltas_per_sec.append(delta/embeddings.shape[0])
                self.data_pool['patch_coords'].append(patch_coords)
                self.data_pool['img_id'].append(img_ids)
                self.data_pool['embedding'].append(embeddings)
                if 'class_id' in batch:
                    if 'class_id' in self.data_pool:
                        self.data_pool['class_id'].append(batch['class_id'].numpy())
                    else:
                        self.data_pool['class_id'] = [batch['class_id'].numpy()]
                n_examples += embeddings.shape[0]
                pbar.set_postfix_str(f"Processed: {n_examples:,}")
                # print(n_examples/self.chunk_size)
                if self.chunk_size is not None and n_examples / self.chunk_size>=part:
                    if self.save_embeddings:
                        #save in different chunks to avoid exceeding RAM
                        postfix = f'part_{part}'
                        self.save_datapool(postfix=postfix)
                        self.reset_data_pool()
                        part+=1
                        if n_examples >= self.max_pool_size:
                            break


                elif self.chunk_size is None and n_examples >= self.max_pool_size:
                    if self.save_embeddings and len(self.data_pool['embedding'])>0:
                        self.save_datapool(postfix=f'part_{part}')
                        self.reset_data_pool()
                    break
            else:
                # loop was not broken -> max pool size not reached
                # save a subset smaller than chunk_size
                if self.save_embeddings and len(self.data_pool['embedding'])>0:
                    self.save_datapool(postfix=f'part_{part}')
                    self.reset_data_pool()

            # TODO (JL): check if this is always the case:
            self.data_pool['embedding'] = np.concatenate(self.data_pool['embedding'])
            self.data_pool['img_id'] = np.concatenate(self.data_pool['img_id'])
            self.data_pool['patch_coords'] = np.concatenate(self.data_pool['patch_coords'])


            # normalize, as this is required for the scann library
            # self.data_pool = {key: np.concatenate(self.data_pool[key], axis=0) for key in self.data_pool}
            print(f'Finish extraction of {n_examples} feature embeddings')
            overall_time_with_loading = overall_start - time.time()
            overall_time_with_loading_per_sample = overall_time_with_loading / n_examples
            overall_time = np.sum(np.asarray(deltas))
            overall_time_per_sample = overall_time / n_examples
            print('='*25, ' Time results ','='*25)
            print(f'Extraction alone took {overall_time} secs = {overall_time/60} mins = {overall_time/3600} hrs')
            print(f'Extraction with loading took {overall_time_with_loading} secs = {overall_time_with_loading / 60} mins = {overall_time_with_loading / 3600} hrs')
            print(f'Extraction-only time per sample: {overall_time_per_sample}')
            print(f'Overall time per sample: {overall_time_with_loading_per_sample}')
            print('='*60)
            if self.save_embeddings and self.chunk_size is None:
                # only save a single file, when chunk size not defined
                self.saved_embeddings = self.save_datapool()
                self.reset_data_pool()
        except Exception as e:
            print(f'Catched {e.__class__.__name__}: {e}, calculating results:')
            print(f'Finish extraction of {n_examples} feature embeddings')
            overall_time_with_loading = overall_start - time.time()
            overall_time_with_loading_per_sample = overall_time_with_loading / n_examples
            overall_time = np.sum(np.asarray(delta))
            overall_time_per_sample = overall_time / n_examples
            print('=' * 25, ' Time results ', '=' * 25)
            print(f'Extraction alone took {overall_time} secs = {overall_time / 60} mins = {overall_time / 3600} hrs')
            print(f'Extraction with loading took {overall_time_with_loading} secs = {overall_time_with_loading / 60} mins = {overall_time_with_loading / 3600} hrs')
            print(f'Extraction-only time per sample: {overall_time_per_sample}')
            print(f'Overall time per sample: {overall_time_with_loading_per_sample}')
            print('=' * 60)

    def get_nn_patches(self, batched_nns):
        nn_patches = []
        for nns in batched_nns:
            current_nn_patches = []
            for nn_id in nns:

                    
                img = self.patch_dset[nn_id]['image']
                if nn_id not in self.data_pool['img_id']:
                    img_id = nn_id
                    patch_coords = [  0,   0, 256, 256]
                else:
                    img_id = self.data_pool['img_id'][nn_id]
                    patch_coords = self.data_pool['patch_coords'][nn_id]
                

                # let patcher crop, so that non-square images
                # can also be handled
                patch = self.patch_dset.get_patch(img, patch_coords)
                if isinstance(patch, np.ndarray):
                    patch = torch.from_numpy(patch)

                current_nn_patches.append(patch)

            current_nn_patches = torch.stack(current_nn_patches, dim=0)
            nn_patches.append(current_nn_patches)

        return torch.stack(nn_patches, dim=0)

    @torch.no_grad()
    def embed(self, batch, is_caption=False):
        if not is_caption:
            if self.gpu:
                batch = batch.cuda()
            if batch.ndim == 5:
                batch = rearrange(batch, 'b n h w c -> (b n) h w c')
            batch = rearrange(batch, 'b h w c -> b c h w')
            batch = batch.to(memory_format=torch.contiguous_format).float()
            bs = batch.shape[0]
        else:
            bs = len(batch)
        return self.retriever(batch).cpu().numpy().reshape(bs, -1)

    def search_kn_for_dset(self, data_config):
        raise NotImplementedError('Wait a little bit')

    def search_k_nearest(self, **kwargs):
        if self.db_engine == 'scann':
            return self.search_k_nearest_scann(**kwargs)
        if self.db_engine == 'faiss':
            return self.search_k_nearest_faiss(**kwargs)

    def search_k_nearest_faiss(self, queries, subsample_rate=None, k=None, is_caption=False,visualize=None, query_embedded=False):
        if not visualize:
            visualize = self.visualize
        if not subsample_rate:
            subsample_rate = self.subsample_rate
        if not query_embedded:
            if isinstance(queries, np.ndarray):
                queries = torch.from_numpy(queries)
            query_embeddings_ = self.embed(queries, is_caption=is_caption)
        else:
            query_embeddings_ = queries
        query_embeddings = query_embeddings_ / np.linalg.norm(query_embeddings_, axis=1)[:, np.newaxis]
        if not k:
            k = self.k

        start = time.time()
        
        print('subsampling', subsample_rate)
        # Privacy amplification by subsampling
        if subsample_rate < 1:
            distances = []
            nns = []
            ntotal = self.searcher.ntotal
            for idx in range(query_embeddings.shape[0]):
                sel = faiss.IDSelectorBatch(subsample_index(dset_size=ntotal, p=subsample_rate))
                query = np.array([query_embeddings[idx]]).astype('float32')
                D, I = self.searcher.search(query, k=k, params=faiss.SearchParametersIVF(sel=sel, nprobe=512))
                distances.append(D[0])
                nns.append(I[0])
        else:
            distances, nns = self.searcher.search(np.array(query_embeddings).astype('float32'), k=k)
        nns = np.array(nns).astype(int)
        print(f'Query time: {time.time() - start}s')
        start = time.time()
        
        
        if hasattr(self.dset, 'stored_embeddings') and self.dset.stored_embeddings:
            out_embeddings = []
            out_img_ids = nns # self.data_pool['img_id'][nns]
            out_pc = np.tile(np.tile([  0,   0, 256, 256], [k, 1]), (self.retriever_bs ,1, 1))
            try:
                for nn_list in nns:
                    out_embeddings.append([self.dset.fetch_embedding(idx) for idx in nn_list])
                out_embeddings = np.array(out_embeddings)    
            except:
                out_embeddings = self.data_pool['embedding'][nns]
        else:
            out_embeddings = self.data_pool['embedding'][nns]
            out_img_ids = self.data_pool['img_id'][nns]
            out_pc = self.data_pool['patch_coords'][nns]
        print(f'Fetching: {time.time()- start}s')
        start = time.time()
        out = {'embeddings': out_embeddings,
               'img_ids': out_img_ids,
               'patch_coords':out_pc,
               'queries': queries,
               'exec_time': time.time() - start,
               'nns': nns,
               'nn_patches': [],
               'q_embeddings': query_embeddings_}
        
        if visualize: #or self.additional_embedders is not None:
            patches = self.get_nn_patches(nns)
            out.update({'nn_patches': patches})
        print(f'Getting Patches: {time.time() - start}s')

        return out


    def search_k_nearest_scann(self, queries, k=None, is_caption=False,visualize=None, query_embedded=False):
        assert self.searcher is not None, 'Cannot search with uninitialized searcher'

        if not query_embedded:
            if isinstance(queries, np.ndarray):
                queries = torch.from_numpy(queries)
            query_embeddings_ = self.embed(queries, is_caption=is_caption)
        else:
            query_embeddings_ = queries
        query_embeddings = query_embeddings_ / np.linalg.norm(query_embeddings_, axis=1)[:, np.newaxis]

        start = time.time()
        nns, distances = self.searcher.search_batched(query_embeddings, final_num_neighbors=k)
        end = time.time()

        out_embeddings = self.data_pool['embedding'][nns]
        out_img_ids = self.data_pool['img_id'][nns]
        out_pc = self.data_pool['patch_coords'][nns]

        out = {'embeddings': out_embeddings,
               'img_ids': out_img_ids,
               'patch_coords':out_pc,
               'queries': queries,
               'exec_time': end - start,
               'nns': nns,
               'q_embeddings': query_embeddings}

        if visualize is None:
            visualize = self.visualize

        if visualize: #or self.additional_embedders is not None:
            patches = self.get_nn_patches(nns)
            out.update({'nn_patches': patches})

            # if self.additional_embedders is not None:
            #     patches = rearrange(torch.from_numpy(patches))
            #     for key in self.additional_embedders:
            #         with torch.no_grad():


        return out

    def search_bruteforce(self, searcher):
        return searcher.score_brute_force().build()

    def search_partioned_ah(self, searcher, dims_per_block, aiq_threshold, reorder_k,
                            partioning_trainsize, num_leaves, num_leaves_to_search):
        return searcher.tree(num_leaves=num_leaves,
                             num_leaves_to_search=num_leaves_to_search,
                             training_sample_size=partioning_trainsize). \
            score_ah(dims_per_block, anisotropic_quantization_threshold=aiq_threshold).reorder(reorder_k).build()

    def search_ah(self, searcher, dims_per_block, aiq_threshold, reorder_k):
        return searcher.score_ah(dims_per_block, anisotropic_quantization_threshold=aiq_threshold).reorder(
            reorder_k).build()

    def train_searcher_faiss(self, 
                       metric=faiss.METRIC_INNER_PRODUCT,
                       searcher_savedir=None,
                       **kwargs):
        if searcher_savedir is None and not self.save_searcher:
            searcher_savedir = self.searcher_savedir
        
        searcher_path = f'{searcher_savedir}/faiss.index'

        if searcher_savedir is not None and os.path.isfile(searcher_path):
            self.searcher = faiss.read_index(searcher_path)
            return

        embeddings = self.data_pool['embedding']
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        print('Train searcher faiss - shape', embeddings.shape)
        embeddings = np.array(embeddings).astype('float32')
        
        embedding_dim = embeddings.shape[-1]
        index = faiss.IndexFlat(embedding_dim, metric)
        
        print('Adding to searcher', embeddings.shape)
        index.add(embeddings)
        index.train(embeddings)
        print('Finished')

        if searcher_savedir is not None and self.save_searcher:
            print(f'Save trained searcher under "{searcher_savedir}"')
            os.makedirs(searcher_savedir, exist_ok=True)
            faiss.write_index(index, searcher_path)
            
        self.searcher = index
    def train_searcher(self, **kwargs):
        print("Searcher DB Engine: ", self.db_engine)
        if self.db_engine == 'scann':
            return self.train_searcher_scann(**kwargs)
        if self.db_engine == 'faiss':
            return self.train_searcher_faiss(**kwargs)
    

    
    
    def train_searcher_scann(self, k=None,
                       metric=None,
                       partioning_trainsize=None,
                       reorder_k=None,
                       # todo tune
                       aiq_thld=0.2,
                       dims_per_block=2,
                       num_leaves=None,
                       num_leaves_to_search=None,
                       searcher_savedir=None):
        if searcher_savedir is None and not self.save_searcher:
            searcher_savedir = self.searcher_savedir

        if searcher_savedir is not None and os.path.isdir(searcher_savedir) and not self.save_searcher:

            if self.searcher is None:
                if k != self.k and k is not None:
                    print(f'Retraining searcher with new k which is {k} (but NOT saving or overriding saved searcher!). default k is {self.k}')
                else:
                    print(f'load pretrained searcher from {searcher_savedir}')
                    self.searcher = scann.scann_ops_pybind.load_searcher(searcher_savedir)
                    print('Finished loading searcher.')
                    return
            else:
                print('Using trained searcher')
                return


        if not k:
            k = self.k

        if not metric:
            metric = self.distance_metric

        # todo tune
        if not reorder_k:
            reorder_k = 2 * k

        # normalize
        embeddings = self.data_pool['embedding'] / np.linalg.norm(self.data_pool['embedding'], axis=1)[:, np.newaxis]
        searcher = scann.scann_ops_pybind.builder(embeddings, k, metric)
        pool_size = self.data_pool['embedding'].shape[0]
        

        print(*(['#'] * 100))
        print('Initializing scaNN searcher with the following values:')
        print(f'k: {k}')
        print(f'metric: {metric}')
        print(f'reorder_k: {reorder_k}')
        print(f'anisotropic_quantization_threshold: {aiq_thld}')
        print(f'dims_per_block: {dims_per_block}')
        print(*(['#'] * 100))
        print('Start training searcher....')
        print(f'N samples in pool is {pool_size}')

        # this reflects the recommended design choices proposed at
        # https://github.com/google-research/google-research/blob/aca5f2e44e301af172590bb8e65711f0c9ee0cfd/scann/docs/algorithms.md
        if pool_size < 2e4:
            print('Using brute force search.')
            self.searcher = self.search_bruteforce(searcher)
        elif 2e4 <= pool_size and pool_size < 1e5:
            print('Using asymmetric hashing search and reordering.')
            self.searcher = self.search_ah(searcher, dims_per_block, aiq_thld, reorder_k)
        else:
            print('Using using partioning, asymmetric hashing search and reordering.')

            if not partioning_trainsize:
                partioning_trainsize = self.data_pool['embedding'].shape[0] // 10
            if not num_leaves:
                num_leaves = int(np.sqrt(pool_size))

            if not num_leaves_to_search:
                num_leaves_to_search = max(num_leaves // 20, 1)

            print('Partitioning params:')
            print(f'num_leaves: {num_leaves}')
            print(f'num_leaves_to_search: {num_leaves_to_search}')
            # self.searcher = self.search_ah(searcher, dims_per_block, aiq_thld, reorder_k)
            self.searcher = self.search_partioned_ah(searcher, dims_per_block, aiq_thld, reorder_k,
                                                     partioning_trainsize, num_leaves, num_leaves_to_search)

        print('Finish training searcher')

        if searcher_savedir is not None and k == self.k:
            print(f'Save trained searcher under "{searcher_savedir}"')
            os.makedirs(searcher_savedir, exist_ok=True)
            self.searcher.serialize(searcher_savedir)

    def load_retriever(self, gpu=True, eval_mode=True):

        if 'ckpt_path' in self.retriever_config.params:
            ckpt = self.retriever_config.params.ckpt_path
            sd = torch.load(ckpt, map_location='cpu')
            sd = sd['state_dict']
        else:
            sd = None

        model = instantiate_from_config(self.retriever_config)
        if sd is not None:
            try:
                model.load_state_dict(sd)
            except RuntimeError:
                # guess we are dealing with a diffusion-wrapper problem
                # hack a little
                new_sd = dict()
                for key in sd:
                    if key.startswith("model."):
                        newkey = "model.diffusion_model." + key[len("model."):]
                    elif key.startswith("model_ema"):
                        newkey = "model_ema.diffusion_model" + key[len("model_ema."):]
                    else:
                        newkey = key
                    new_sd[newkey] = sd[key]
                m, u = model.load_state_dict(new_sd, strict=False)
                print("missing keys:")
                print(m)
                print("Unexpected Keys")
                print(u)
        if gpu:
            model.cuda()
        if eval_mode:
            model.eval()
        return model