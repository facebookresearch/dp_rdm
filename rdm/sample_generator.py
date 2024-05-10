#!/usr/bin/env python3
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
from clip import tokenize
import numpy as np
import torchvision
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config
import pickle

from PIL import Image
from rdm.util import custom_to_pil
from rdm.data.cifar10 import Cifar10Validation

from experiments_util import (
    mscoco_processor,
)
from experiments_util import (
    CIFAR10_PROMPTS,
    MSCOCO_PROMPTS,
    SHUTTERSTOCK_PROMPTS,
    SHUTTERSTOCK_VALIDATION_PATH,
)


# Retrieval database names:
cifar_validation_data = Cifar10Validation(
    size=32, root="/datasets01/cifar-pytorch/11222017/"
)


def setup_model(o):
    config = OmegaConf.load(o.config_path)
    config.model.params.retrieval_cfg.params.load_patch_dataset = True
    config.model.params.retrieval_cfg.params.gpu = True
    # Load state dict
    pl_sd = torch.load(o.ckpt_path, map_location="cpu")
    
    # Initialize model
    model = instantiate_from_config(config.model)

    # Apply checkpoint
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    print("Loaded model.")

    # Eval mode
    model = model.eval()
    device = torch.ones(1).cuda().device

    print("device", device)
    model.to(device)

    return model, device

def format_args(
    model,
    sigma,
    captions,
    k_nn,
    ddim_steps,
    guidance_scale,
    query_embedding_interpolation,
    aggregate,
    subsample_rate=1.0,
    return_nns=False,
    public_retrieval=True,
    **kwargs,
):
    tokenized = tokenize(captions).to(model.device)
    clip = model.retriever.retriever.model
    q_emb = clip.encode_text(tokenized).cpu()
    del tokenized

    visualize_nns = True
    if not return_nns:
        visualize_nns = False

    sample_args = dict(
        query=q_emb,
        query_embedded=True,
        use_weights=False,
        ddim=True,
        visualize_nns=visualize_nns,
        unconditional_retro_guidance_label=0.0,
        k_nn=k_nn,
        unconditional_guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
        query_embedding_interpolation=query_embedding_interpolation,
        aggregate=aggregate,
        public_retrieval=public_retrieval,
        sigma=sigma,
        return_nns=return_nns,
        subsample_rate=subsample_rate,
    )
    print(sample_args)
    return sample_args

class SampleGenerator:
    def __init__(self, model, device, opts):
        self.model = model
        self.device = device
        self.o = opts

    def generate_samples(self):
        o = self.o
        pathlib.Path(o.sample_folder).mkdir(parents=True, exist_ok=True)

        n_recs = o.rec_end - o.rec_start
        n_batches = n_recs // o.batch_size
        all_captions = []
        save_pickle_path = f"{o.sample_folder}/params.{o.rec_start:05}.pickle"
        if os.path.exists(save_pickle_path):
            print("SKIPPING - ", save_pickle_path)
            return

        for batch in range(n_batches):
            i_start = batch * o.batch_size + o.rec_start
            i_end = (batch + 1) * o.batch_size + o.rec_start
            captions = []
            img_ids = []
            for idx in range(i_start, i_end):
                for subdir in ["val", "syn", "knn"]:
                    pathlib.Path(f"{o.sample_folder}/{subdir}").mkdir(
                        parents=True, exist_ok=True
                    )
                img_id, caption = self.generate_validation_sample(o, self.model, self.device, idx)
                all_captions.append((img_id, caption))
                captions.append(caption)
                img_ids.append(img_id)

            save_path = lambda idx: f"{o.sample_folder}/syn/{img_ids[idx]:012}.png"

            skip_batch = False
            count = 0
            for idx, _ in enumerate(img_ids):
                if os.path.exists(save_path(idx)):
                    count += 1
            if count == len(img_ids):
                skip_batch = True

            if skip_batch:
                print(f"[SKIP] \t batch {batch} already generated")
            else:
                logs = self.model.sample_with_query(
                    **format_args(model=self.model, captions=captions, **vars(o))
                )
                for idx, im in enumerate(logs["query_samples"]):
                    pil_im = custom_to_pil(im)
                    pil_im.save(save_path(idx))
                    if "retro_nns" in logs.keys():
                        p = custom_to_pil(logs["retro_nns"][idx].detach().cpu())
                        p.save(f"{o.sample_folder}/knn/{img_ids[idx]:012}.png")
        hparams = {"params": vars(o), "all_captions": all_captions}
        hparams["params"]["delta"] = o.dp_delta

        print(save_pickle_path)
        with open(save_pickle_path, "wb") as f:
            pickle.dump(hparams, f)


    def generate_validation_sample(self, o, model, device, idx):
        if o.validation_dataset == 'cifar10':
            img_id, caption = self.generate_cifar_val(o, idx)
        if o.validation_dataset == 'mscoco':
            img_id, caption = self.generate_mscoco_val(o, model, device, idx)
        if o.validation_dataset == 'shutterstock':
            img_id, caption = self.generate_shutterstock_val(o, idx)
        return img_id,caption

    def generate_shutterstock_val(self, o, idx):
        img_id, caption = SHUTTERSTOCK_PROMPTS[idx]
        img_id = int(img_id)
        Image.open(
                        f"{SHUTTERSTOCK_VALIDATION_PATH}/{img_id:012}.jpg"
                    ).convert("RGB").save(
                        f"{o.sample_folder}/val/{img_id:012}.png"
                    )
        
        return img_id,caption

    def generate_mscoco_val(self, o, model, device, idx):
        img_id, caption = MSCOCO_PROMPTS[idx]
        image = Image.open(
                        f"{o.mscoco_validation_basepath}/COCO_val2014_{img_id:012}.jpg"
                    ).convert("RGB")
        image = mscoco_processor(image=np.array(image).astype(np.uint8))["image"]
        image = (
                        torchvision.transforms.ToTensor()(image).to(device).unsqueeze(0)
                        * 2
                        - 1
                    )
        encoded = model.encode_first_stage(image)
        print(f"writing... {img_id}, caption: {caption}")
        custom_to_pil(model.decode_first_stage(encoded)[0]).save(
                        f"{o.sample_folder}/val/{img_id:012}.png"
                    )
        return img_id, caption

    def generate_cifar_val(self, o, idx):
        sample = cifar_validation_data[idx]
        caption = CIFAR10_PROMPTS[sample["target"]]
        image = Image.fromarray(((sample['image'] + 1)* 127.5).astype(np.uint8), 'RGB')
        image.save(
                        f"{o.sample_folder}/val/{idx:012}.png"
                    )
        img_id = idx
        return img_id, caption
