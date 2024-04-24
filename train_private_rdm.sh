RDM_PATH=/private/home/lebensold/dev/private-augmented-retrieval
# for resume of no agg, no noise
#CFG=in64_sig0.2_agg
#CFG=in64_nonoise_k1
#CFG=in64_sig0.2_agg

#CFG=in64fb_sig0.1_agg
#CFG=in64fb_sig0.2_agg


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
# job: 21646745
#train_model "in64fb_sig0_noagg"
resume_training_model "in64fb_sig0_noagg" \
"/checkpoint/lebensold/rdm/21646745/" \
"/checkpoint/lebensold/rdm/21646745/2024-01-01T15-42-37_rdm_in64fb_sig0_noagg_mclip_norm/checkpoints/last.ckpt"


#train_model "in64fb_sig0.1_agg"
resume_training_model "in64fb_sig0.1_agg" \
"/checkpoint/lebensold/rdm/21646746/" \
"/checkpoint/lebensold/rdm/21646746/2024-01-01T15-42-44_rdm_in64fb_sig0.1_agg_mclip_norm/checkpoints/last.ckpt"
