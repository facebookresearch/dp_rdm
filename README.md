# DP-RDM: Differentially-Private Retrieval-Augmented Diffusion Models
[Paper (arxiv)](https://arxiv.org/abs/2403.14421)


Text-to-image diffusion models tend to suffer from sample-level memorization, possibly reproducing near-perfect replica of images that they are trained on, which may be undesirable. To remedy this issue, we develop the first differentially private (DP) retrieval-augmented generation algorithm that is capable of generating high-quality image samples while providing provable privacy guarantees. Specifically, we assume access to a text-to-image diffusion model trained on a small amount of public data, and design a DP retrieval mechanism to augment the text prompt with samples retrieved from a private retrieval dataset. Our \emph{differentially private retrieval-augmented diffusion model} (DP-RDM) requires no fine-tuning on the retrieval dataset to adapt to another domain, and can use state-of-the-art generative models to generate high-quality image samples while satisfying rigorous DP guarantees. For instance, when evaluated on MS-COCO, our DP-RDM can generate samples with a privacy budget of $\epsilon=10$, while providing a $3.5$ point improvement in FID compared to public-only retrieval for up to $10,000$ queries. 

# Getting Started
1. Setup your conda environment:
```
 conda env create --file environment.yml 
```
2. Activate the environment:
```
conda activate dp_rdm
```

3. Download the first stage models:
Download the first stage models using the following script:
```
$ scripts/download_first_stages.sh
```

4. Download metaclip:
```
wget https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt models/metaclip/b16_400m.pt
```


## 1. DP-RDM Privacy Analysis
The `experiments_utils.calc_eps` function calculates the differential privacy budget, denoted as epsilon , and identifies the optimal order of Rényi divergence (alpha) using the Rényi Differential Privacy (RDP) accountant method. It inputs the standard deviation of Gaussian noise, subsample rate, the number of nearest neighbors (k), the total number of queries, and the target delta of differential privacy. It then adjusts the noise multiplier based on global sensitivity, evaluates the privacy loss over specified orders of alpha, and returns the minimum epsilon and the corresponding alpha. 

See `privacy_analysis.ipynb` for example calculations of epsilon with respect to dataset size, subsample rate, noise magnitude and number of neighbors.

## 2. DP-RDM Adapt Training

# Training and Sample Generation: DP-RDM Modifications
Below are the key touchpoints for DP-RDM: private sample generation, privacy-adapted training and privacy analysis.

The `rdm.util.aggregate_and_noise` function in Python aggregates and perturbs the embeddings, then adds scaled Gaussian noise where the noise magnitude is randomly adjusted between 0 and a maximum value, and then replicates the resulting embedding. If aggregation is not enabled, it directly adds similarly scaled and adjusted Gaussian noise to the original embeddings. This function is useful for embedding manipulation during training processes, potentially for data augmentation or regularization purposes.


### Preparation of retrieval database
* Download a database of your choice from above.
* Precompute nearest neighbors for a given query dataset:
    * Create new config for QueryDataset under `configs/query_datasets` (see template for creation)
    * Start nn extraction with `python scripts/search_neighbors.py -rc <path to retrieval_config> -qc <path to query config> -s <query dataset split> -bs <batch size> -w <n_workers>`, e.g. `python scripts/search_neighbors.py --rconfig configs/dataset_builder/imagenet_fb.yaml --qc configs/query_datasets/imagenet_fb.yaml -s train -n`

To train a new model, make sure you have ImageNet face blurred in the correct path defined in `rdm.data.imagenet.ImageNetValidationFaceBlurred` and `rdm.data.imagenet.ImageNetTrainFaceBlurred`.

Then run 
```
./train_private_rdm.sh
```

## 3. Private Retrieval for Sample Generation

The function `rdm.util.aggregate_and_noise_query` takes four parameters: `aggregate`, `r_enc`, `k_nn`, and `noise_magnitude`. 

1. **Parameter Initialization and Device Setting**:
   - The function starts by identifying the device of the tensor `r_enc` to ensure all operations occur on the same computational device.

3. **Aggregation and Noise Addition (if `aggregate` is True)**:
   - The embeddings are summed across the second dimension (assuming batch-first ordering in tensors), and the result is reshaped by adding a new dimension. This is equivalent to aggregating embeddings.
   - The aggregated result is then normalized by dividing by `k_nn`, which represents the number of embeddings that were aggregated.
   - Gaussian noise is added to the aggregated embeddings. The noise is scaled by the `noise_magnitude` and added directly to `r_enc`.   
   - The function returns the potentially modified embeddings `r_enc` and the logging dictionary.

Once you have a trained model, you can generate private samples. 

See `sample_generation.ipynb` for usage example.

# 4. Generating Embeddings
DP-RDM relies on a private retrieval dataset generated from metaclip embeddings. First, generate embeddings for the entire dataset using `wds_to_clip.py` as example code. Then you need to create a fast index with Faiss. See `wds_build_faiss.py` for example code.



# RDM
The code components of DP-RDM were built as an extension of Retrieval-Augmented Diffusion Models.

[**Retrieval-Augmented Diffusion Models**](https://arxiv.org/abs/2204.11824)<br/>

Our codebase depends on [Retrieval-Augmented Diffusion Models](https://github.com/CompVis/retrieval-augmented-diffusion-models).

## BibTeX

```
@article{lebensold2024dp,
  title={DP-RDM: Adapting Diffusion Models to Private Domains Without Fine-Tuning},
  author={Lebensold, Jonathan and Sanjabi, Maziar and Astolfi, Pietro and Romero-Soriano, Adriana and Chaudhuri, Kamalika and Rabbat, Mike and Guo, Chuan},
  year={2024},
  doi={https://doi.org/10.48550/arXiv.2403.14421},
  url={https://arxiv.org/abs/2403.14421}
}
```