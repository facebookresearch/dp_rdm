# DP-RDM: Differentially-Private Retrieval-Augmented Diffusion Models
Text-to-image diffusion models tend to suffer from sample-level memorization, possibly reproducing near-perfect replica of images that they are trained on, which may be undesirable. To remedy this issue, we develop the first differentially private (DP) retrieval-augmented generation algorithm that is capable of generating high-quality image samples while providing provable privacy guarantees. Specifically, we assume access to a text-to-image diffusion model trained on a small amount of public data, and design a DP retrieval mechanism to augment the text prompt with samples retrieved from a private retrieval dataset. Our \emph{differentially private retrieval-augmented diffusion model} (DP-RDM) requires no fine-tuning on the retrieval dataset to adapt to another domain, and can use state-of-the-art generative models to generate high-quality image samples while satisfying rigorous DP guarantees. For instance, when evaluated on MS-COCO, our DP-RDM can generate samples with a privacy budget of $\epsilon=10$, while providing a $3.5$ point improvement in FID compared to public-only retrieval for up to $10,000$ queries. 

# Generating Embeddings
See `shutterstock_1_wds_to_clip.py` for example code.

# Training and Sample Generation: DP-RDM Modifications
Below are the key touchpoints for DP-RDM: private sample generation, privacy-adapted training and privacy analysis.

## 1. Private Retrieval for Sample Generation

The function `rdm.util.aggregate_and_noise_query` takes four parameters: `aggregate`, `r_enc`, `k_nn`, and `noise_magnitude`. 

1. **Parameter Initialization and Device Setting**:
   - The function starts by identifying the device of the tensor `r_enc` to ensure all operations occur on the same computational device.

3. **Aggregation and Noise Addition (if `aggregate` is True)**:
   - The embeddings are summed across the second dimension (assuming batch-first ordering in tensors), and the result is reshaped by adding a new dimension. This is equivalent to aggregating embeddings.
   - The aggregated result is then normalized by dividing by `k_nn`, which represents the number of embeddings that were aggregated.
   - Gaussian noise is added to the aggregated embeddings. The noise is scaled by the `noise_magnitude` and added directly to `r_enc`.   
   - The function returns the potentially modified embeddings `r_enc` and the logging dictionary.

## 2. DP-RDM Adapt Training

The `rdm.util.aggregate_and_noise` function in Python manipulates a set of embeddings by optionally averaging and then replicating them, or directly adding variable noise to them. If aggregation is enabled, it averages the embeddings, adds scaled Gaussian noise where the noise magnitude is randomly adjusted between 0 and a maximum value, and then replicates the resulting embedding. If aggregation is not enabled, it directly adds similarly scaled and adjusted Gaussian noise to the original embeddings. This function is useful for embedding manipulation during training processes, potentially for data augmentation or regularization purposes.

## 3. DP-RDM Privacy Analysis
The `experiments_utils.calc_eps` function calculates the differential privacy budget, denoted as epsilon , and identifies the optimal order of Rényi divergence (alpha) using the Rényi Differential Privacy (RDP) accountant method. It inputs the standard deviation of Gaussian noise, subsample rate, the number of nearest neighbors (k), the total number of queries, and the target delta of differential privacy. It then adjusts the noise multiplier based on global sensitivity, evaluates the privacy loss over specified orders of alpha, and returns the minimum epsilon and the corresponding alpha. 

# RDM
The code components of DP-RDM were built as an extension of Retrieval-Augmented Diffusion Models.

[**Retrieval-Augmented Diffusion Models**](https://arxiv.org/abs/2204.11824)<br/>
[Andreas Blattmann](https://github.com/ablattmann)\*,
[Robin Rombach](https://github.com/rromb)\*,
[Kaan Oktay](https://github.com/kaanoktay)\,
[Jonas Müller](https://github.com/jenuk),
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution


## Comments
- Our codebase depends on [DP-RDM](https://github.com/CompVis/retrieval-augmented-diffusion-models)

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