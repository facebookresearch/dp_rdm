"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os

import kornia
import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from ldm.models.autoencoder import (AutoencoderKL, IdentityFirstStage,
                                    VQModelInterface)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config, isimage, ismap, log_txt_as_img
from ldm.modules.ema import LitEma

from rdm.models.diffusion.ddim import DDIMSampler
from rdm.util import aggregate_and_noise, aggregate_and_noise_query

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class RETRODiffusionWrapper(pl.LightningModule):
    """
    for now, drop options regarding old code and only use cross-attention
    """

    def __init__(self, diffusion_wrapper,concat=False):
        super().__init__()
        self.concat = concat
        if concat:
            print(f'WARNING: {self.__class__.__name__} is concatenating conditionings in sequence dimension. Assuming same embedding dimension')
            self.diffusion_model = diffusion_wrapper
        else:
            self.diffusion_model = diffusion_wrapper.diffusion_model
        self.conditioning_key = diffusion_wrapper.conditioning_key
        self.wrapper_conditioning_key = diffusion_wrapper.conditioning_key
        print(f"{self.__class__.__name__}: Wrapping diffusion model for RETRO training. "
              f"For multimodal data, conditionings will be chained in a list "
              f"and all fed into the 'SpatialTransformer' via different cross-attention "
              f"blocks.")

    def forward(self, x, t, c_crossattn: list = None):
        key = 'c_crossattn' if self.concat else 'context'
        out = self.diffusion_model(x, t, **{key:c_crossattn})
        return out

# Diffusion Model: 
class MinimalRETRODiffusion(LatentDiffusion):
    """main differences to base class:
        - dataloading to build the conditioning
        - concat the base conditioning (e.g. text) and the new retro-conditionings
        - maybe adopt log_images
    """
    def __init__(self, k_nn, query_key, retrieval_encoder_cfg, nn_encoder_cfg=None, query_encoder_cfg=None,
                 nn_key='retro_conditioning', retro_noise=False, retrieval_cfg=None,retro_conditioning_key=None,
                 learn_nn_encoder=False, nn_memory=None,
                 n_patches_per_side = 1, resize_patch_size=None,
                 searcher_path=None, retro_concat=False,
                 p_uncond=0., guidance_vex_shape=None,
                 aggregate=False, k_nn_max_noise=0.,
                 normalized_embeddings=True,
                 public_retrieval_cfg=None,
                 *args, **kwargs):
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', [])
        use_ema = kwargs.pop('use_ema',True)
        unet_config = kwargs.get('unet_config',{})
        
        
        super().__init__(*args, **kwargs)
        self.normalized_embeddings = normalized_embeddings
        self.aggregate = aggregate # do we aggregate instead of sample k_NN? 
        self.k_nn_max_noise = k_nn_max_noise  # max noise added to the k_nn vector during training
        print(f"MinimalRETRODiffusion - agg: {self.aggregate}, noise: {self.k_nn_max_noise}, normalized embeddings: {self.normalized_embeddings} ")
        self.k_nn = k_nn  # number of neighbors to retrieve
        self.query_key = query_key  # e.g. "clip_txt_emb"
        self.nn_key = nn_key
        # backwards compat

        self.model = RETRODiffusionWrapper(self.model,concat=retro_concat)  # adopts the forward pass to retro-style cross-attn training
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.searcher_path = searcher_path

        self.use_memory = nn_memory is not None and os.path.isfile(nn_memory)

        self.retriever = None
        self.public_retriever = None
        self.learn_nn_encoder = learn_nn_encoder
        self.init_nn_encoder(nn_encoder_cfg)  # TODO attention while restoring, do not want to overwrite
        self.resize_nn_patch_size = resize_patch_size
        print('retrieval_cfg', retrieval_cfg)
        self.init_retriever(retrieval_cfg)
        print('public_retrieval_cfg', public_retrieval_cfg)
        self.init_public_retriever(public_retrieval_cfg)
        # most likely a transformer, can apply masking within this models' forward pass
        self.conditional_retrieval_encoder = query_encoder_cfg is not None
        if self.conditional_retrieval_encoder and 'cross_attend' not in retrieval_encoder_cfg.params:
            print(
                f'WARNING: intending to train query conditioned retrieval encoder without cross attention, adding option to cfg...')
            retrieval_encoder_cfg.params['cross_attend'] = True
        self.retrieval_encoder = instantiate_from_config(
            retrieval_encoder_cfg)  # TODO attention while restoring, do not want to overwrite
        self.init_query_encoder(query_encoder_cfg)

        self.retro_noise = retro_noise

        self.n_patches_per_side = n_patches_per_side

        self.use_retriever_for_retro_cond = not self.retriever is None and not self.use_memory

        self.p_uncond = p_uncond
        # if p_uncond > 0:
        if guidance_vex_shape is None:
            guidance_vex_shape = (unet_config.params.context_dim,)
            print(f'Setting guiding vex shape to (,) (assuming clip nn encoder)')


        self.get_unconditional_guiding_vex(tuple(guidance_vex_shape))
        # else:
        #     ignore_keys += ['unconditional_guidance_vex']

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

    def init_query_encoder(self, cfg):
        if not self.conditional_retrieval_encoder:
            return
        self.query_encoder = instantiate_from_config(cfg)
        print(f'Using {self.query_encoder.__class__.__name__} as query encoder.')


    def init_nn_encoder(self, cfg):
        if not cfg:
            self.nn_encoder = None
            self.resize_nn_patches = False
            return
        self.resize_nn_patches = cfg.params.pop('resize_nn_patches', False)
        if cfg == '__is_first_stage__':
            self.learn_nn_encoder = False
            print("Using first stage also as nn_encoder.")
            self.nn_encoder = self.first_stage_model
            self.resize_nn_patches = True
        else:
            self.nn_encoder = instantiate_from_config(cfg)
            additional_info = 'LEARNABLE' if self.learn_nn_encoder else 'FIXED'
            print(f'Loading {additional_info} nn_encoder of type {self.nn_encoder.__class__.__name__}')
            if not self.learn_nn_encoder:
                self.nn_encoder.train = disabled_train
                for param in self.nn_encoder.parameters():
                    param.requires_grad = False

        cfg.params['resize_nn_patches'] = self.resize_nn_patches

    @rank_zero_only
    def train_searcher(self):
        print("training searcher...")
        self.retriever.train_searcher()
        print("done training searcher")



    @rank_zero_only
    def init_retriever(self, cfg):
        if not cfg:
            self.retriever = None
            return
        # this is the nearest neighbor searcher
        self.retriever = instantiate_from_config(cfg)
        self.retriever.train = disabled_train
        for param in self.retriever.retriever.parameters():
            param.requires_grad = False

    @rank_zero_only
    def init_public_retriever(self, cfg):
        if not cfg:
            self.public_retriever = None
            return
        # this is the nearest neighbor searcher
        self.public_retriever = instantiate_from_config(cfg)
        self.public_retriever.train = disabled_train
        for param in self.public_retriever.retriever.parameters():
            param.requires_grad = False

        print("training public searcher...")
        self.public_retriever.train_searcher()
        print("done!")

    def resize_img_batch(self,img_batch,size):
        return kornia.geometry.resize(img_batch,size=(size,size),align_corners=True)

    @torch.no_grad()
    def encode_with_fixed_nn_encoder(self,nns2encode,shape = None):
        if self.resize_nn_patches:
            resize_size = self.resize_nn_patch_size if self.resize_nn_patch_size else self.first_stage_model.encoder.resolution
            nns2encode = self.resize_img_batch(nns2encode, resize_size)
        if isinstance(self.nn_encoder, VQModelInterface):
            # for backwards compatibility
            if self.model.wrapper_conditioning_key in ['concat'] or (self.model.wrapper_conditioning_key == 'hybrid' and self.model.retro_conditioning_key == 'concat'):
                nn_encodings = self.nn_encoder.encode(nns2encode)
            else:
                assert shape is not None, 'Need to give \'em a shape'
                bs, nptch, k = shape[:3]
                nn_encodings = self.nn_encoder.encode(nns2encode).reshape(bs, nptch * k, -1)
        else:
            # NNEnoder is assumed to do reshaping
            nn_encodings = self.nn_encoder.encode(nns2encode)

        return nn_encodings


    @torch.no_grad()
    def get_retro_conditioning(self, batch, return_patches=False, bs=None,
                               use_learned_nn_encoder=False,k_nn=None, use_retriever=None):
        """
        given x, compute its nearest neighbors via the self.retriever module
        """
        if k_nn is None:
            k_nn = self.k_nn
        output = dict()
        if bs is not None:
            batch = {key: batch[key][:bs] for key in batch}

        if use_retriever is None:
            use_retriever = self.use_retriever_for_retro_cond

    
        # retro conditioning is expected to have shape  (bs,n_query_patches,k,embed_dim)
        if self.nn_encoder is None:
            # use retriever embeddings
            nns = batch[self.nn_key]
            # reshape appropriately for transformer

            output[self.nn_key] = rearrange(nns, 'b n k d -> b (n k) d').to(torch.float)
        else:
            # use nn_patches with defined retrieval embedder model
            assert self.nn_encoder.device == self.device
            # bs, nptch, k = batch['nn_patches'].shape[:3]
            nn_patches = rearrange(batch['nn_patches'], 'b n k h w c -> (b n k) c h w').to(self.device).to(
                torch.float)
            if not self.learn_nn_encoder or use_learned_nn_encoder:
                output[self.nn_key] = self.encode_with_fixed_nn_encoder(nn_patches,batch['nn_patches'].shape)

        if return_patches and 'nn_patches' in batch:
            output['image_patches'] = rearrange(batch['nn_patches'], 'b n k h w c -> (b n) k h w c')
    
        return output  # as a sequence of neighbor representations, sorted by increasing distance.

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        r = self.get_retro_conditioning(batch,return_patches=self.learn_nn_encoder)
        if self.p_uncond > 0.:
            mask = torch.distributions.Bernoulli(torch.full((x.shape[0],),self.p_uncond)).sample().to(self.device).bool()
            uncond_signal = self.get_unconditional_conditioning(shape=r[self.nn_key].shape)
            r[self.nn_key] = torch.where(repeat(mask,'b -> b 1 1 '), uncond_signal,r[self.nn_key])
        loss = self(x, c, r)
        return loss



    def forward(self, x, c, r, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        qs = {}

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

        noise = torch.randn_like(x) # what happens if sigma < 1? 
        
        if self.conditional_retrieval_encoder:
            x_noisy = self.q_sample(x, t, noise=noise)
            q = self.query_encoder(x_noisy)
            qs = {'context': q}

        # if self.learn_nn_encoder:
        #     assert self.nn_encoder is not None, 'If you wanna learn nn_encoders, please define such a thing :)'
        #     # NNEnoder is assumed to do reshaping
        #     r[self.nn_key] = self.nn_encoder.encode(rearrange(r['image_patches'],'(b n) k h w c -> (b n k) c h w',
        #                                                       b=x.shape[0],n= self.n_patches_per_side**2,k=self.k_nn).to(self.device).to(torch.float))

        r_enc = self.retrieval_encoder(r[self.nn_key], **qs)  # can also be initialized to the identity

        
        #r_enc = r_enc / torch.norm(r_enc, dim=1)[:, np.newaxis]
        #r_enc = r_enc / r_enc.norm(dim=-1, keepdim=True)

        print('BEFORE', r_enc, r_enc.shape, 'NORM', torch.norm(r_enc, dim=-1)) 
        r_enc = r_enc / torch.norm(r_enc, dim=-1, keepdim=True)
        
        print('AFTER NORM', r_enc, r_enc.shape, 'NORM', torch.norm(r_enc, dim=-1)) 
        r_enc = aggregate_and_noise(aggregate=self.aggregate, 
                                    r_enc=r_enc, 
                                    k_nn=self.k_nn,
                                    k_nn_max_noise=self.k_nn_max_noise)
        
        print('AFTER AGG', r_enc, r_enc.shape, 'NORM', torch.norm(r_enc, dim=-1)) 

        #print('RETRO NOISE? ', self.retro_noise) # this is False
        if self.retro_noise:
            r_enc = self.q_sample(r_enc,t)
        
        # c is None!
        if c is not None:
            if self.model.wrapper_conditioning_key == 'hybrid':
                non_retro_ck = 'crossattn' if self.model.retro_conditioning_key == 'concat' else 'concat'
                c = {'c_'+non_retro_ck: c if isinstance(c,list) else [c],
                     'c_'+self.model.retro_conditioning_key: r_enc if isinstance(r_enc,list) else [r_enc]}
            else:
                c = [r_enc, c]
        else:
            if not isinstance(r_enc,list):
                c = [r_enc]
            else:
                c = r_enc

        return self.p_losses(x, c, t, noise=noise, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.wrapper_conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)
        return x_recon

    @rank_zero_only
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., 
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, display_patches=True, memsize=None, 
                   **kwargs):
        if memsize is None:
            memsize = [100]
        use_ddim = ddim_steps is not None
        log = dict()
        z, c, x, xrec, xc = self.get_input(batch,
                                           self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)

        # retrieval-specific hacks
        rdir = self.get_retro_conditioning(batch, return_patches=display_patches, bs=N,
                                           use_learned_nn_encoder=self.learn_nn_encoder)
        display_patches &= 'image_patches' in rdir
        if display_patches:
            r, img_patches = rdir[self.nn_key], rdir["image_patches"]
        else:
            r = rdir[self.nn_key]

        if self.conditional_retrieval_encoder:
            qs = {'context': self.query_encoder(z)}
        else:
            qs = {}

        r_enc = self.retrieval_encoder(r, **qs)
        r_enc = r_enc / torch.norm(r_enc, dim=-1, keepdim=True)
        r_enc = aggregate_and_noise_query(self.aggregate, r_enc, self.k_nn, self.k_nn_max_noise)
        
        if c is not None:
            c = [r_enc, c]
        else:
            if not isinstance(r_enc,list):
                c = [r_enc]
            else:
                c = r_enc

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec

        if display_patches:
            # plot neighbors
            # shape of img_patches: b n h w c
            grid = rearrange(img_patches, 'b n h w c -> (b n) c h w')
            grid = make_grid(grid, nrow=img_patches.shape[1], normalize=True)
            log["neighbors"] = 2. * grid - 1.

        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta,**kwargs)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                                             quantize_denoised=True,**kwargs)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                 ddim_steps=ddim_steps, x0=z[:N], mask=mask,**kwargs)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                                 ddim_steps=ddim_steps, x0=z[:N], mask=mask,**kwargs)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row
        return log

    def get_unconditional_guiding_vex(self,vector_shape):
        # if not hasattr(self, 'unconditional_guidance_vex'):
            # bs, vector_shape = shape[0], shape[1:]


        print('Initializing unconditional guidance vector')
        # if self.p_uncond == 0.:
        init_vec = torch.randn(vector_shape, device=self.device)
        # else:
        #     init_vec = torch.zeros((vector_shape,),device=self.device)
        if self.p_uncond == 0.:
            self.register_buffer('unconditional_guidance_vex', init_vec, persistent=True)
        else:
            self.register_parameter('unconditional_guidance_vex', torch.nn.Parameter(init_vec))

    @torch.no_grad()
    def get_unconditional_conditioning(self, shape, unconditional_guidance_label=None,k_nn=None, ignore_knn=False):

        if k_nn is None:
            k_nn = self.k_nn

        bs, vector_shape = shape[0], shape[-1]

        if not hasattr(self, 'unconditional_guidance_vex'):
            self.get_unconditional_guiding_vex((vector_shape,))

        if unconditional_guidance_label is not None:
            # unconditional guidance label defines norm of vector
            uncond_signal = self.unconditional_guidance_vex / torch.linalg.norm(self.unconditional_guidance_vex.flatten()) * unconditional_guidance_label

            if uncond_signal.shape[0] != self.k_nn and not ignore_knn:
                uncond_signal = torch.stack([uncond_signal] * k_nn, dim=0)

            uncond_signal = torch.stack([uncond_signal]*bs,dim = 0)
        else:
            uncond_signal = torch.stack([self.unconditional_guidance_vex]*bs,dim = 0)

        print(uncond_signal.shape)

        return uncond_signal

    @torch.no_grad()
    def sample_with_query(self, **kwargs):
        query = self.build_query(**kwargs)
        return self.sample_with_cached_query(**query, **kwargs)

    
    @torch.no_grad()
    def build_query(self, query, visualize_nns, k_nn, subsample_rate, return_nns, aggregate, sigma, query_embedding_interpolation, public_retrieval, **kwargs):
        if self.retriever is not None and self.retriever.searcher is None:
            self.train_searcher()
        is_caption = isinstance(query, list)
        if k_nn is None:
            k_nn = self.k_nn
        if isinstance(query, torch.Tensor):
            query = query.cpu().numpy()

        nn_dict = self.retriever.search_k_nearest(queries=query,visualize=visualize_nns, k=k_nn,
                                                  is_caption=is_caption,
                                                  subsample_rate=subsample_rate,
                                                  query_embedded=True)
        out = dict()
        q_emb = torch.from_numpy(nn_dict['q_embeddings'])
        r_emb = torch.from_numpy(nn_dict['embeddings'])
        out['r_emb_before'] = r_emb

        if self.normalized_embeddings:            
            r_emb = r_emb / torch.norm(r_emb, dim=-1, keepdim=True)
            q_emb = q_emb / torch.norm(q_emb, dim=-1, keepdim=True)
        
        out['r_emb_before_agg'] = r_emb
        if public_retrieval and query_embedding_interpolation < 1.:
            if self.public_retriever is None:
                assert "public retriever not defined and required"
            

            public_nn_dict = self.public_retriever.search_k_nearest(queries=query,visualize=visualize_nns, k=k_nn,
                                                  is_caption=is_caption,
                                                  subsample_rate=1.,
                                                  query_embedded=True)
            public_r_emb = torch.from_numpy(public_nn_dict['embeddings'])

            if self.normalized_embeddings:
                public_r_emb = public_r_emb / torch.norm(public_r_emb, dim=-1, keepdim=True)

            q_emb = public_r_emb
        else:
            q_emb = torch.concat([v.repeat(k_nn, 1).unsqueeze(0) for v in q_emb])
        r_emb, logging = aggregate_and_noise_query(aggregate, r_emb, k_nn, sigma)
        out['q_emb'] = q_emb
        out['r_emb_after_agg'] = r_emb
        out['aggregate_logging'] = logging
        
        r_emb = r_emb.to(self.device).float()
        q_emb = q_emb.to(self.device).float()

        # trade-off retrieval dataset VS query embedding
        out['retro_cond'] = query_embedding_interpolation * r_emb + (1 - query_embedding_interpolation ) * q_emb

        out = self.extract_nearest_neighbors(k_nn, return_nns, nn_dict, out)
        return out
    
    def extract_nearest_neighbors(self, k_nn, return_nns, nn_dict, out):
        if return_nns:
            nn_patches = nn_dict['nn_patches'][:, :k_nn]
            
            # if caption based and caption shall be repeated, there are no visual neighbors
            n_per_row = nn_patches.shape[1]
            nn_patches = rearrange(nn_patches, 'b k h w c -> b k c h w').to(self.device)
            out['retro_nns'] = [make_grid(grid, nrow=n_per_row, normalize=True) * 2 - 1 for grid in nn_patches]
            return out
        return out
    
    @torch.no_grad()
    def sample_with_cached_query(self, retro_cond, k_nn, **kwargs):
        c = self.retrieval_encoder(retro_cond)
        # label is 0
        bs = c.shape[0]
        c_unconditional_guidance = self.get_unconditional_conditioning(c.shape, unconditional_guidance_label=0., k_nn=k_nn).float()
        with self.ema_scope("Plotting"):
            samples, _ = self.sample_log(cond=c, 
                                         batch_size=bs, unconditional_conditioning=c_unconditional_guidance, **kwargs)

        x_samples = self.decode_first_stage(samples)
        out = dict() 
        out["query_samples"] = x_samples
        return out
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps,
                   custom_shape=None, del_sampler=False, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            if custom_shape is None:
                shape = (self.channels, self.image_size, self.image_size)
            else:
                shape = custom_shape
            ddim_steps=kwargs.pop('S',ddim_steps)
            verbose = kwargs.pop('verbose',False)

            samples, intermediates = ddim_sampler.sample(S=ddim_steps, batch_size=batch_size,
                                                         shape=shape, conditioning=cond, verbose=verbose, **kwargs)
            if del_sampler:
                del ddim_sampler

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)
        
        
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        params = params + list(
            self.retrieval_encoder.parameters())  # adding the encoder of retrieval embeddings/patches
        if self.conditional_retrieval_encoder:
            params += list(self.query_encoder.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_nn_encoder:
            print(f"{self.__class__.__name__}: Also optimizing nn_encoder params!")
            params = params + list(self.nn_encoder.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt