# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

RDM_PATH=/private/home/lebensold/dev/dp_rdm

train_model() {
    cd $RDM_PATH
    cfg=$1;
    python train_private_rdm_submitit.py \
        --ngpus 8 \
        --nodes 5 \
        --partition learnlab \
        --use_volta32 \
        --base configs/${cfg}.yaml \
        --train \
        --scale_lr false \
        --name rdm_${cfg}_mclip_norm
}

resume_training_model() {
    cd $RDM_PATH
    cfg=$1;
    job_dir=$2;
    last_ckpt=$3;
    python train_private_rdm_submitit.py \
        --ngpus 8 \
        --nodes 5 \
        --partition learnlab \
        --use_volta32 \
        --base configs/${cfg}.yaml \
        --train \
        --scale_lr false \
        --job_dir $job_dir \
        --resume $last_ckpt
}
train_model "in64fb_sig0.05_agg"