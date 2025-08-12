import torch
import math
import numpy as np
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import DynamicCache
from src.builder import BUILDER
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len, device=order.device),
    ).bool()
    return masking


class SkyworkUnipic(nn.Module):
    def __init__(self, vae, vae_scale, llm, mar, tokenizer, prompt_template, siglip2):
        super().__init__()
        # VAE
        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)
        self.vae_scale = vae_scale

        # LLM
        self.llm = BUILDER.build(llm)
        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.image_token_idx = self.tokenizer.convert_tokens_to_ids("<image>")
        
        self.prompt_template = prompt_template

        # MAR
        self.mar = BUILDER.build(mar)
        # projection layers
        self.proj_in = build_mlp(
            hidden_size=self.mar.encoder_embed_dim,
            projector_dim=self.llm.config.hidden_size,
            z_dim=self.llm.config.hidden_size,
        )
        self.proj_out = build_mlp(
            hidden_size=self.llm.config.hidden_size,
            projector_dim=self.llm.config.hidden_size,
            z_dim=self.mar.encoder_embed_dim,
        )

        # siglip
        self.siglip2 = BUILDER.build(siglip2)
        self.siglip2_proj = build_mlp(
            hidden_size=1152,
            projector_dim=self.llm.config.hidden_size,
            z_dim=self.llm.config.hidden_size,
        )

    @property
    def llm_model(self):
        return self.llm.model

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    @property
    def gen_seq_len(self):
        return self.mar.seq_len

    @property
    def token_embed_dim(self):
        return self.vae.embed_dim * (self.mar.patch_size**2)

    @torch.no_grad()
    def encode(self, x):
        posterior = self.vae.encode(x)
        z = posterior.mode().mul_(self.vae_scale)
        z = rearrange(
            z,
            "b c (m p) (n q) -> b m n (c p q)",
            p=self.mar.patch_size,
            q=self.mar.patch_size,
        )

        return z

    @torch.no_grad()
    def decode(self, z):
        z /= self.vae_scale
        z = rearrange(
            z,
            "b m n (c p q) -> b c (m p) (n q)",
            p=self.mar.patch_size,
            q=self.mar.patch_size,
        )

        x = self.vae.decode(z)
        return x

    def prepare_forward_input(
        self,
        x,
        inputs_embeds=None,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
    ):
        b, l, _ = x.shape
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones(b, l)], dim=1
        )
        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids[position_ids < 0] = 0

        # import pdb; pdb.set_trace()

        # prepare context
        if past_key_values is not None:
            inputs_embeds = x
            position_ids = position_ids[:, -l:]
        else:
            if inputs_embeds is None:
                input_ids = input_ids.to(self.device)
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([inputs_embeds, x], dim=1)

        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

    def extract_visual_feature(self, x, mask=None, detach=False):
        b, m, n, _ = x.shape
        x = x.view(b, m * n, -1)
        # x: b mn c
        if mask is None:
            mask = torch.zeros_like(x[..., 0])
        null_embeds = self.mar.fake_latent.expand(x.shape[0], -1)
        x_enc = self.mar.forward_mae_encoder(x, mask, null_embeds, image_shape=(m, n))

        z_enc = self.proj_in(x_enc)
        # Move buffers to the end of the image sequence
        z_enc = torch.cat(
            [z_enc[:, self.mar.buffer_size :], z_enc[:, : self.mar.buffer_size]], dim=1
        )

        if detach:
            x_enc = x_enc.detach()
            z_enc = z_enc.detach()

        return x_enc, z_enc

    def forward_mae_encoder(self, x, mask, detach=False, **context):
        b, m, n, _ = x.shape
        x_enc, z_enc = self.extract_visual_feature(x, mask=mask, detach=detach)
        inputs = self.prepare_forward_input(x=z_enc, **context)
        output = self.llm_model(**inputs, return_dict=True)

        z_llm = output.last_hidden_state[:, -z_enc.shape[1] :]

        # move buffers back to the start of the image sequence
        z_llm = torch.cat(
            [z_llm[:, -self.mar.buffer_size :], z_llm[:, : -self.mar.buffer_size]],
            dim=1,
        )

        # residual learning
        x_enc = x_enc + self.proj_out(z_llm)

        return x_enc

    @staticmethod
    def curtail_cache(past_key_values, cur_len):
        for past_key_values_ in past_key_values:
            keys, values = past_key_values_
            keys.data = keys.data[:, :, :cur_len]
            values.data = values.data[:, :, :cur_len]

    @torch.no_grad()
    def prepare_text_conditions(self, prompt, cfg_prompt="Generate an image."):
        all_prompts = [
            self.prompt_template["INSTRUCTION"].format(input=prompt),
            self.prompt_template["INSTRUCTION"].format(input=cfg_prompt),
        ]

        input_ids = [
            self.tokenizer.encode(p, add_special_tokens=True, return_tensors="pt")[0]
            for p in all_prompts
        ]
        valid_lens = [len(input_ids_) for input_ids_ in input_ids]
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
        attention_mask = torch.zeros_like(input_ids).bool()
        for i in range(len(input_ids)):
            attention_mask[i, : valid_lens[i]] = True

        return dict(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )

    @torch.no_grad()
    def sample(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="constant",
        temperature=1.0,
        progress=False,
        mask=None,
        past_key_values=None,
        image_shape=None,
        x_con=None,
        **kwargs,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        bsz = attention_mask.shape[0]
        if cfg != 1.0:
            assert bsz % 2 == 0

        if image_shape is None:
            m = n = int(self.gen_seq_len**0.5)
        else:
            m, n = image_shape

        if mask is None:
            mask = torch.ones(bsz, m * n, device=self.device, dtype=self.dtype)
        else:
            mask = mask.view(bsz, m * n)
        tokens = torch.zeros(
            bsz, m * n, self.token_embed_dim, device=self.device, dtype=self.dtype
        )
        orders = self.mar.sample_orders(bsz, seq_len=m * n)
        if cfg != 1.0:
            orders[bsz // 2 :] = orders[: bsz // 2]

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)

        # past key values can be prepared outside (usually in multi-turn editing)
        if past_key_values is None:
            output = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                position_ids=None,
                past_key_values=DynamicCache.from_legacy_cache(),
                return_dict=True,
                use_cache=True,
            )
            past_key_values = output.past_key_values

        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            x_enc = self.forward_mae_encoder(
                tokens.view(bsz, m, n, -1),
                mask.to(self.dtype),
                past_key_values=past_key_values,
                # inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            # import pdb; pdb.set_trace()
            self.curtail_cache(past_key_values, inputs_embeds.shape[1])
            # import pdb; pdb.set_trace()

            z = self.mar.forward_mae_decoder(
                x_enc, mask.to(self.dtype), image_shape=(m, n), x_con=x_con
            )

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(m * n * mask_ratio)]).to(self.device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).to(self.device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, m * n).to(self.device)
            if cfg != 1.0:
                mask_next[bsz // 2 :] = mask_next[: bsz // 2]
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            # if not cfg == 1.0:
            #     mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (m * n - mask_len[0]) / (m * n)
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.mar.diffloss.sample(
                z, temperature, cfg_iter
            ).to(self.dtype)
            # if not cfg == 1.0:
            #     sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
            #     mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            if cfg != 1.0:
                cur_tokens[bsz // 2 :] = cur_tokens[: bsz // 2]
            tokens = cur_tokens.clone()

        pred = self.decode(tokens.view(bsz, m, n, -1))

        if cfg != 1.0:
            pred = pred[: bsz // 2]
        return pred
