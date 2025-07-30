#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config
from PIL import Image
from src.builder import BUILDER
from src.datasets.utils import crop2square
from torch.nn.utils.rnn import pad_sequence

import os
import torch

import os
import torch


def preprocess_image(image: Image.Image, image_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Process PIL image to normalized tensor [1,C,H,W]."""
    img = crop2square(image)
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = 2 * arr - 1
    tensor = torch.from_numpy(arr).to(dtype=dtype)
    return rearrange(tensor, "h w c -> 1 c h w")


class EditInferencer:
    def __init__(self, config_path: str, checkpoint_path: str, image_size: int):
        # 1) Build model
        self.cfg = Config.fromfile(config_path)
        self.model = BUILDER.build(self.cfg.model).eval().cuda().to(torch.bfloat16)

  
        # 2) Load model weights
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # print(f"Unexpected parameters: {unexpected}")
        # print(f"Missed parameters: {missing}")

        self.image_size = image_size

        special_tokens_dict = {'additional_special_tokens': ["<image>", ]}
        num_added_toks = self.model.tokenizer.add_special_tokens(special_tokens_dict)

        self.image_token_idx = self.model.tokenizer.encode("<image>", add_special_tokens=False)[-1]
        print(f"Image token: {self.model.tokenizer.decode(self.image_token_idx)}")

    def edit_image(
        self, 
        source_image: Image.Image, 
        prompt: str, 
        num_iter: int = 48, 
        cfg: float = 3.0,
        cfg_prompt: str = "Repeat this image.",
        cfg_schedule: str = "constant",
        temperature: float = 0.85,
        grid_size: int = 1
    ) -> Image.Image:
        """Edit single image based on prompt."""
        # 1) Preprocess source image
        img_tensor = preprocess_image(
            source_image, 
            self.image_size,
            dtype=self.model.dtype
        ).to(self.model.device)
        
        # 2) Encode image and extract features
        with torch.no_grad():
            x_enc = self.model.encode(img_tensor)
            x_con, z_enc = self.model.extract_visual_feature(x_enc)
        
        # 3) Prepare text prompts
        m = n = self.image_size // 16
        image_length = m * n + 64
        
        if hasattr(self.cfg.model, 'prompt_template'):
            prompt_str = self.cfg.model.prompt_template['INSTRUCTION'].format(
                input="<image>\n" + prompt.strip()
            )
            cfg_prompt_str = self.cfg.model.prompt_template['INSTRUCTION'].format(
                input="<image>\n" + cfg_prompt.strip()
            )
        else:
            prompt_str = f"<image>\n{prompt.strip()}"
            cfg_prompt_str = f"<image>\n{cfg_prompt.strip()}"
        
        # Replace <image> token with multiple tokens
        prompt_str = prompt_str.replace('<image>', '<image>' * image_length)
        cfg_prompt_str = cfg_prompt_str.replace('<image>', '<image>' * image_length)
        
        # 4) Tokenize and prepare inputs
        input_ids = self.model.tokenizer.encode(
            prompt_str, add_special_tokens=True, return_tensors='pt')[0].cuda()
        
        if cfg != 1.0:
            null_input_ids = self.model.tokenizer.encode(
                cfg_prompt_str, add_special_tokens=True, return_tensors='pt')[0].cuda()
            attention_mask = pad_sequence(
                [torch.ones_like(input_ids), torch.ones_like(null_input_ids)],
                batch_first=True, padding_value=0).to(torch.bool)
            input_ids = pad_sequence(
                [input_ids, null_input_ids],
                batch_first=True, padding_value=self.model.tokenizer.eos_token_id)
        else:
            input_ids = input_ids[None]
            attention_mask = torch.ones_like(input_ids).to(torch.bool)
        
        # 5) Prepare embeddings
        if cfg != 1.0:
            z_enc = torch.cat([z_enc, z_enc], dim=0)
            x_con = torch.cat([x_con, x_con], dim=0)
        
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.model.llm.config.hidden_size)
        inputs_embeds[input_ids == self.image_token_idx] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != self.image_token_idx] = self.model.llm.get_input_embeddings()(
            input_ids[input_ids != self.image_token_idx]
        )
        
        # 6) Repeat for grid sampling
        bsz = grid_size ** 2
        x_con = torch.cat([x_con] * bsz)
        if cfg != 1.0:
            inputs_embeds = torch.cat([
                inputs_embeds[:1].expand(bsz, -1, -1),
                inputs_embeds[1:].expand(bsz, -1, -1),
            ])
            attention_mask = torch.cat([
                attention_mask[:1].expand(bsz, -1),
                attention_mask[1:].expand(bsz, -1),
            ])
        else:
            inputs_embeds = inputs_embeds.expand(bsz, -1, -1)
            attention_mask = attention_mask.expand(bsz, -1)
        
        # 7) Sampling
        samples = self.model.sample(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_iter=num_iter,
            cfg=cfg,
            cfg_schedule=cfg_schedule,
            temperature=temperature,
            progress=False,
            image_shape=(m, n),
            x_con=x_con
        )
        
        # 8) Convert to PIL Image
        samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=grid_size, n=grid_size)
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        out = samples.to("cpu", torch.uint8).numpy()
        return Image.fromarray(out)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="UniPic model editing inference on single input")
    parser.add_argument("config", help="Model config file path")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, default="data/sample.png")
    parser.add_argument("--prompt", type=str, default="Replace the stars with the candle.")
    parser.add_argument("--output", default="results", help="Output image path")
    parser.add_argument("--image_size", type=int, default=1024, help="Image size for processing")
    parser.add_argument("--num_iter", type=int, default=32, help="Number of sampling iterations")
    parser.add_argument("--cfg", type=float, default=3, help="Classifier-free guidance scale")
    parser.add_argument("--cfg_prompt", type=str, default="Repeat this image.", help="Prompt for CFG null condition")
    parser.add_argument("--cfg_schedule", type=str, default="constant", choices=["constant", "linear", "cosine"], help="CFG schedule type")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--grid_size", type=int, default=1, help="Grid size for sampling multiple variants")
    args = parser.parse_args()

    # Initialize inferencer on this GPU
    inferencer = EditInferencer(config_path = args.config, checkpoint_path = args.checkpoint,image_size =args.image_size)
    src_img = Image.open(args.image)
    out_img = inferencer.edit_image(
        source_image=src_img,
        prompt=args.prompt,
        num_iter=args.num_iter,
        cfg=args.cfg,
        cfg_prompt=args.cfg_prompt,
        cfg_schedule=args.cfg_schedule,
        temperature=args.temperature,
        grid_size=args.grid_size
    )
    out_img.save(args.output)

if __name__ == "__main__":
    main()

