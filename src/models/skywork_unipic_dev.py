import torch
import torch.nn.functional as F
from torch.nn.modules.module import T
from mmengine.model import BaseModel
from torch.autograd.function import Function
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
import os
#from .skywork_unipic import SkyworkUnipic
from .skywork_unipic_siglip import SkyworkUnipic
from xtuner.utils import IMAGE_TOKEN_INDEX
import torch.distributed as dist
import json
from einops import rearrange


def _load_state_dict_with_ds(module_to_load, state_dict, start_prefix="", strict=True):
    try:
        import deepspeed
    except ImportError:
        raise ImportError("deepspeed is not installed. Please install deepspeed to use this feature.")
    
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []
    missing_keys = []
    unexpected_keys = []

    def load(module: torch.nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False)
                )
                params_to_gather = [
                    named_parameters[k]
                    for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(
                        params_to_gather, modifier_rank=0
                    ):
                        if deepspeed.comm.get_rank() == 0:
                            module._load_from_state_dict(*args)
        else:
            module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(module_to_load, state_dict, start_prefix)
    if len(missing_keys) > 0:
        print_log(f"[WARNING] Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print_log(f"[WARNING] Unexpected keys: {unexpected_keys}")
    if error_msgs:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                module_to_load.__class__.__name__, "\n\t".join(error_msgs)
            )
        )


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class SkyworkUnipicDev(SkyworkUnipic, BaseModel):
    def __init__(
        self,
        grad_scale=0.1,
        loss_weights=None,
        pretrained_pth=None,
        mar_path=None,
        siglip_proj_path=None,
        freeze_llm=False,
        freeze_mar=False,
        freeze_mar_decoder=False,
        freeze_siglip_proj=False,
        gradient_checkpointing=True,
        **kwargs,
    ):
        if loss_weights is None:
            loss_weights = {
                "image2text": 0.01,
                "text2image": 1.0,
                "image_edit": 1.0,
                "contrastive": 0.1,
            }
        super().__init__(**kwargs)
        
        self.grad_scale = grad_scale
        self.loss_weights = loss_weights
        self.pretrained_pth = pretrained_pth
        self.mar_path = mar_path
        self.siglip_proj_path = siglip_proj_path

        # 判断分布式 rank
        rank = dist.get_rank() if dist.is_initialized() else 0

        # === 加载预训练权重 ===
        if pretrained_pth:
            self.load_hf_weights(
                skywork_unipic_ckpt=pretrained_pth,
                siglip_proj_path=siglip_proj_path,
                mar_path=mar_path
            )

        # === 冻结模块 ===
        if freeze_llm:
            self.llm.requires_grad_(False)
        if freeze_mar:
            self.mar.requires_grad_(False)
        if freeze_mar_decoder:
            # 仅冻结 MAR 解码器部件
            for param in self.mar.decoder_embed.parameters():
                param.requires_grad = False
            for block in self.mar.decoder_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            for param in self.mar.decoder_norm.parameters():
                param.requires_grad = False
            if isinstance(self.mar.decoder_pos_embed_learned, torch.nn.Parameter):
                self.mar.decoder_pos_embed_learned.requires_grad = False
            if isinstance(self.mar.diffusion_pos_embed_learned, torch.nn.Parameter):
                self.mar.diffusion_pos_embed_learned.requires_grad = False
        if freeze_siglip_proj:
            self.siglip2_proj.requires_grad_(False)

        # === 梯度检查点 ===
        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

    
    def load_hf_weights(self,
                        skywork_unipic_ckpt: str = None,
                        siglip_proj_path: str = None,
                        mar_path: str = None):
        """统一加载 SkyworkUnipic（可选） + SigLIP2 + MAR"""
        device = "cpu"
        state_dict = {}

        def _print_load_result(module_name, missing, unexpected):
            print_log(f"[INFO] Loaded {module_name}. missing={len(missing)}, unexpected={len(unexpected)}")

        # === SkyworkUnipic 主模型（可选） ===
        if skywork_unipic_ckpt:
            print_log(f"[INFO] Loading SkyworkUnipic checkpoint from: {skywork_unipic_ckpt}")
            # 加载 checkpoint（支持文件或目录）
            if os.path.isfile(skywork_unipic_ckpt):
                skywork_unipic_state = torch.load(skywork_unipic_ckpt, map_location=device)
            else:
                idx = os.path.join(skywork_unipic_ckpt, "pytorch_model.bin.index.json")
                if os.path.exists(idx):
                    with open(idx, 'r') as f:
                        index = json.load(f)
                    skywork_unipic_state = {}
                    for shard in sorted(set(index["weight_map"].values())):
                        shard_path = os.path.join(skywork_unipic_ckpt, shard)
                        skywork_unipic_state.update(torch.load(shard_path, map_location=device))
                else:
                    bin_path = os.path.join(skywork_unipic_ckpt, "pytorch_model.bin")
                    skywork_unipic_state = torch.load(bin_path, map_location=device)

            # 删除 SkyworkUnipic checkpoint 中可能带的 MAR pos_embed，避免覆盖

            # for key in [
            #     "mar.encoder_pos_embed_learned",
            #     "mar.decoder_pos_embed_learned",
            #     "mar.diffusion_pos_embed_learned"
            # ]:
            #     if key in skywork_unipic_state:
            #         print_log(f"[INFO] Dropping `{key}` from SkyworkUnipic checkpoint")
            #         del skywork_unipic_state[key]
            model_dict = self.state_dict()
            
            filtered_checkpoint = {}
            shape_mismatch_keys = []

            for k, v in skywork_unipic_state.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        filtered_checkpoint[k] = v
                    else:
                        shape_mismatch_keys.append((k, v.shape, model_dict[k].shape))

            missing, unexpected = self.load_state_dict(filtered_checkpoint, strict=False)
            # 打印不匹配的 key 及其形状
            if shape_mismatch_keys:
                print("以下 key 因形状不匹配被跳过：")
                for k, checkpoint_shape, model_shape in shape_mismatch_keys:
                    print(f"  - {k}:")
                    print(f"    checkpoint 中的形状: {checkpoint_shape}")
                    print(f"    当前模型的形状: {model_shape}")
            else:
                print("所有 key 形状匹配，未跳过任何参数")

            # missing, unexpected = self.load_state_dict(skywork_unipic_state, strict=False)
            _print_load_result("SkyworkUnipic", missing, unexpected)
        else:
            print_log("[INFO] Skipping SkyworkUnipic checkpoint loading")

        # === SigLIP2 权重 ===
        if siglip_proj_path:
            print_log(f"[INFO] Loading SigLIP2 weights from: {siglip_proj_path}")
            siglip_state = torch.load(
                siglip_proj_path, map_location="cpu", weights_only=False
            )
            # 如果 checkpoint 是 {"model": {...}}
            if isinstance(siglip_state, dict) and "model" in siglip_state:
                siglip_state = siglip_state["model"]
            missing, unexpected = self.siglip2_proj.load_state_dict(
                siglip_state, strict=False
            )
            _print_load_result("SigLIP2", missing, unexpected)
        else:
            print_log("[INFO] No SigLIP2 checkpoint provided, skipping")

        # === MAR 权重 ===
        if mar_path:
            print_log(f"[INFO] Loading MAR weights from: {mar_path}")
            mar_state = torch.load(mar_path, map_location="cpu", weights_only=False)
            # 兼容 model_ema or model dict

            if isinstance(mar_state, dict) and "model_ema" in mar_state:
                mar_state = mar_state["model_ema"]

            elif isinstance(mar_state, dict) and "model" in mar_state:
                mar_state = mar_state["model"]
            
            
            # 如果 key 带有 “mar.” 前缀，批量去掉
            if any(k.startswith("mar.") for k in mar_state):
                filtered_mar = {
                    k.replace("mar.", "", 1): v
                    for k, v in mar_state.items()
                    if k.startswith("mar.")
                }
            else:
                filtered_mar = mar_state

            missing, unexpected = self.mar.load_state_dict(
                filtered_mar, strict=False
            )
            _print_load_result("MAR", missing, unexpected)
        else:
            print_log("[INFO] No MAR checkpoint provided, skipping")

        return state_dict



    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.mar.gradient_checkpointing_enable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict = {k: v for k, v in state_dict.items()
                      if 'vae.' not in k}

        return state_dict

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        return self

    def text2image_loss(self, data_dict):
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)   # b m n c
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m*n, -1)

        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        x_enc = self.forward_mae_encoder(x, mask, input_ids=input_ids,
                                         attention_mask=attention_mask)
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))

        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss
    
    def image2text_loss(self, data_dict):
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        labels = data_dict['labels'].to(self.device)

        pixel_values = data_dict.get('pixel_values', None)
        # print("pixel_values batch:", pixel_values.shape)
        # print("input_ids batch:", input_ids.shape)
        if pixel_values is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            _, z_null = self.extract_visual_feature(
                torch.zeros(1, 16, 16, self.token_embed_dim,
                            dtype=self.dtype, device=self.device)
            )
            loss_null = z_null.mean() * 0.0
            print(f"No image found in this batch!", flush=True)
        else:
            x = pixel_values.to(dtype=self.dtype, device=self.device)
            x = self.encode(x)  # b m n c
            _, z_enc = self.extract_visual_feature(x)

            if self.grad_scale is not None:
                z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

            inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
            
            self.tokenizer.add_tokens(["<image>"], special_tokens=True)
            IMAGE_TOKEN_INDEX = self.tokenizer.convert_tokens_to_ids("<image>")
            # print(f"IMAGE_TOKEN_INDEX: {IMAGE_TOKEN_INDEX}")
            img_tokens = (torch.tensor(input_ids) == IMAGE_TOKEN_INDEX).sum().item()
            # print(f"[校验日志] input_ids长度: {len('input_ids')}, 图像token出现次数: {img_tokens}\n")

            inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
            inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
                input_ids[input_ids != IMAGE_TOKEN_INDEX])
            loss_null = 0.0

        output = self.llm_model(inputs_embeds=inputs_embeds,
                                attention_mask=attention_mask,
                                return_dict=True)

        last_hidden_state = output.last_hidden_state[:, :-1]
        labels = labels[:, 1:]
        last_hidden_state = last_hidden_state[labels >= 0]
        labels = labels[labels >= 0]
        logits = self.llm.get_output_embeddings()(last_hidden_state)

        loss_i2t = F.cross_entropy(input=logits, target=labels)

        return loss_i2t + loss_null
    
    # def image_edit_loss(self, data_dict):
    #     # 1. 图像前向：拼 batch 并编码到视觉特征
    #     x_src = data_dict['pixel_values_src'].to(dtype=self.dtype, device=self.device)  # 源图像批次，shape=[b_src, C, H, W]
    #     x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)          # 编辑图像批次，shape=[b_edit, C, H, W]
    #     print_log(f"[DEBUG image_edit_loss] x_src.shape = {x_src.shape}, x.shape = {x.shape}", level="WARNING")

    #     # b_edit 应该 >= b_src
    #     assert x.shape[0] >= x_src.shape[0], \
    #         f"编辑批次大小 ({x.shape[0]}) 必须 >= 源图像批次大小 ({x_src.shape[0]})"

    #     # 拼接并一次性编码
    #     x_all = torch.cat([x_src, x], dim=0)          # shape=[b_src + b_edit, C, H, W]
    #     x_all = self.encode(x_all)                   # shape=[b_src + b_edit, m, n, c]
    #     # 分割回源/编辑两部分
    #     x_src_enc, x_enc = x_all.split([x_src.shape[0], x.shape[0]], dim=0)
    #     # x_src_enc.shape=[b_src, m, n, c], x_enc.shape=[b_edit, m, n, c]

    #     # 2. 提取视觉特征：x_con 用于 decoder 条件，z_src 用于填充文本中的 <image> token
    #     x_con, z_src = self.extract_visual_feature(x_src_enc)
    #     if self.grad_scale is not None:
    #         x_con = _ScaleGradient.apply(x_con, self.grad_scale)
    #         z_src = _ScaleGradient.apply(z_src, self.grad_scale)
    #     # z_src.shape = [b_src, m*n, C]

    #     # 3. 文本条件分支：构造 inputs_embeds
    #     attention_mask = data_dict['attention_mask'].to(self.device)  # shape=[b_edit, seq_len]
    #     input_ids      = data_dict['input_ids'].to(self.device)       # shape=[b_edit, seq_len]
    #     b_edit, seq_len = input_ids.shape
    #     hidden_size     = self.llm.config.hidden_size

    #     # 先准备一个全 0 的 inputs_embeds
    #     inputs_embeds = z_src.new_zeros(b_edit, seq_len, hidden_size)  # shape=[b_edit, seq_len, hidden_size]

    #     # 找到所有 <image> token 位置的 mask
    #     mask_imgpos = (input_ids == IMAGE_TOKEN_INDEX)                  # bool tensor [b_edit, seq_len]

    #     # 需要将单个 z_src 展开成 b_edit 份，再按 mask_imgpos 填入
    #     # 1) expand：把 z_src 从 [b_src, m*n, C] → [b_edit, m*n, C]
    #     #    （一般 b_src=1，所以就是复制那一份）
    #     z_src_rep = z_src.expand(b_edit, -1, -1)                        # [b_edit, m*n, C]
    #     # 2) flatten：将二维展开到一维，对应 mask_imgpos.sum() 个位置
    #     flat_z = z_src_rep.flatten(0, 1)                               # [b_edit*m*n, C]

    #     # **重要检查**：保证 mask_imgpos 中 True 的数量 == flat_z.shape[0]
    #     img_tokens_count = mask_imgpos.sum().item()
    #     assert img_tokens_count == flat_z.shape[0], \
    #         f"<image> token 数 ({img_tokens_count}) 不等于视觉特征数 ({flat_z.shape[0]})"

    #     # 填充视觉 token 对应位置
    #     inputs_embeds[mask_imgpos] = flat_z

    #     # 剩下的位置用文本 embedding
    #     txt_pos = ~mask_imgpos
    #     txt_embeddings = self.llm.get_input_embeddings()(input_ids[txt_pos])
    #     inputs_embeds[txt_pos] = txt_embeddings

    #     # 4. MAE-style 重建分支：在 decoder 前注入 inputs_embeds 与 attention_mask
    #     b, m, n, c = x_enc.shape
    #     gt = x_enc.view(b, m*n, c)                                     # 作为重建目标
    #     orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
    #     mask   = self.mar.random_masking(x_enc.flatten(1, 2), orders)

    #     # 带条件的 encoder forward
    #     x_enc_out = self.forward_mae_encoder(
    #         x_enc,
    #         mask,
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask
    #     )
    #     # decoder 重建
    #     z_dec = self.mar.forward_mae_decoder(
    #         x_enc_out,
    #         mask,
    #         image_shape=(m, n),
    #         x_con=x_con
    #     )
    #     # 计算损失
    #     loss = self.mar.forward_loss(z=z_dec, target=gt, mask=mask)
    #     return loss


    # def image_edit_loss_vae(self, data_dict):
    #     """
    #     计算图像编辑任务的损失。
    #     参考图(x_src)的特征直接作为条件(x_con)送入解码器，不参与编码器重建。
    #     编码器(encoder)仅在目标图(x_tgt)上进行掩码重建，并接收文本和参考图的上下文信息。
    #     """
    #     # === 步骤 1: 读入数据 ===
    #     x_src = data_dict['pixel_values_src'].to(self.device).to(self.dtype)
    #     x_tgt = data_dict['pixel_values'].to(self.device).to(self.dtype)
    #     attention_mask = data_dict['attention_mask'].to(self.device)
    #     input_ids = data_dict['input_ids'].to(self.device)
    #     # IMG_TOKEN_INDEX = self.tokenizer.convert_tokens_to_ids("<image>")
    #     B = x_tgt.shape[0]

    #     # === 步骤 2: 处理参考图 (Reference Image) ===
    #     # VAE编码，不计算梯度
    #     with torch.no_grad():
    #         z_src_latent = self.encode(x_src)  # [B, m, n, token_dim]

    #     # 将VAE潜变量转换为解码器条件(x_con)和LLM输入(z_src_buf)
    #     # 这一步实现了 "参考图潜变量 -> 解码器" 的直接通路
    #     x_con, z_src_buf = self.vae_latent_to_decoder_feature(z_src_latent)
    #     # x_con:    [B, 4096, enc_dim] -> 用于解码器
    #     # z_src_buf: [B, 4160, llm_dim] -> 用于LLM

    #     # === 步骤 3: 构建LLM的输入 (inputs_embeds) ===
    #     # 结合文本指令(input_ids)和参考图特征(z_src_buf)
    #     _, T = input_ids.shape
    #     H_llm = self.llm.config.hidden_size
    #     inputs_embeds = torch.zeros(B, T, H_llm, device=self.device, dtype=z_src_buf.dtype)

    #     # 填充<image> token和文本token的嵌入
    #     inputs_embeds[input_ids == IMG_TOKEN_INDEX] = z_src_buf.flatten(0, 1)
    #     # input_ids 为33280
    #     # z_src_buf.flatten(0, 1) 为33792 为什么 会比input_ids 多512个呢？
    #     inputs_embeds[input_ids != IMG_TOKEN_INDEX] = self.llm.get_input_embeddings()(
    #         input_ids[input_ids != IMG_TOKEN_INDEX]
    #     )

    #     # === 步骤 4: 处理目标图 (Target Image) 并进行编码器前向传播 ===
    #     # VAE编码目标图，不计算梯度
    #     with torch.no_grad():
    #         z_tgt_latent = self.encode(x_tgt)  # [B, m, n, token_dim]

    #     # 为目标图潜变量创建掩码(mask)以进行MAE重建
    #     B, m, n, token_dim = z_tgt_latent.shape
    #     patch_tokens_tgt = z_tgt_latent.view(B, m * n, token_dim)  # 作为重建的目标
    #     orders = self.mar.sample_orders(bsz=B, seq_len=m * n)
    #     mask = self.mar.random_masking(patch_tokens_tgt, orders)

    #     # **核心**: 编码器只处理目标图(z_tgt_latent)的可见部分，并接收LLM的上下文
    #     x_enc = self.forward_mae_encoder(
    #         z_tgt_latent,  # 目标图潜变量
    #         mask,
    #         detach=False,
    #         inputs_embeds=inputs_embeds,  # 包含文本和参考图信息的上下文
    #         attention_mask=attention_mask
    #     )

    #     # === 步骤 5: 解码器重建 ===
    #     # 解码器使用编码器的输出(x_enc)和参考图的特征(x_con)来重建完整的潜在表示
    #     z_pred = self.mar.forward_mae_decoder(
    #         x_enc,
    #         mask,
    #         image_shape=(m, n),
    #         x_con=x_con  # ★ 参考图特征直接作用于此
    #     )

    #     # === 步骤 6: 计算损失 ===
    #     loss = self.mar.forward_loss(
    #         z=z_pred,
    #         target=patch_tokens_tgt,
    #         mask=mask
    #     )
    #     return loss
    
    def image_edit_loss_contrastive(self, data_dict):
        # Step 1: 获取图像特征
        x_src = data_dict['pixel_values_src'].to(dtype=self.dtype, device=self.device)
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        assert len(x_src) >= len(x)
        x_src, x = self.encode(torch.cat([x_src, x])).split([len(x_src), len(x)], dim=0)

        # Step 2: 文本输入部分
        attention_mask = data_dict['attention_mask'].to(self.device)
        input_ids = data_dict['input_ids'].to(self.device)

        x_con, z_src = self.extract_visual_feature(x_src)
        if self.grad_scale is not None:
            z_src = _ScaleGradient.apply(z_src, self.grad_scale)
            x_con = _ScaleGradient.apply(x_con, self.grad_scale)

        inputs_embeds = z_src.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        # IMAGE_TOKEN_INDEX = self.tokenizer.convert_tokens_to_ids("<image>")
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_src.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX]
        )

        # Step 3: 计算 reconstruction loss
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m * n, -1)
        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)
        x_enc = self.forward_mae_encoder(x, mask,
                                        inputs_embeds=inputs_embeds,
                                        attention_mask=attention_mask)
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n), x_con=x_con)
        rec_loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

        # Step 4: Contrastive loss between repeat and edit
        # 假设 batch 是偶数，按 (repeat, edit) 对排列
        z_src_flat = z_src.mean(dim=1)  # [B, D] 全局池化
        z_src_flat = F.normalize(z_src_flat, dim=-1)

        repeat_z = z_src_flat[::2]  # even index
        edit_z = z_src_flat[1::2]   # odd index

        logits = torch.matmul(edit_z, repeat_z.T) / 0.07  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)

        return rec_loss + self.loss_weights.get("contrastive") * contrastive_loss

    def image_edit_loss(self, data_dict):
        # Multi-turn editing is also supported
        x_src = data_dict['pixel_values_src'].to(dtype=self.dtype, device=self.device)
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        # print_log(f"[DEBUG] x_src.shape = {x_src.shape}, x.shape = {x.shape}")

        # assert len(x_src) >= len(x)
        x_cat = torch.cat([x_src, x], dim=0)
        x_src, x = self.encode(x_cat).split([len(x_src), len(x)], dim=0)

        # Prepare context, including source images and instructions
        attention_mask = data_dict['attention_mask'].to(self.device)
        input_ids = data_dict['input_ids'].to(self.device)

        x_con, z_src = self.extract_visual_feature(x_src)
        if self.grad_scale is not None:
            z_src = _ScaleGradient.apply(z_src, self.grad_scale)
            x_con = _ScaleGradient.apply(x_con, self.grad_scale)

        inputs_embeds = z_src.new_zeros(*input_ids.shape, self.llm.config.hidden_size)

        self.tokenizer.add_tokens(["<image>"], special_tokens=True)

        IMAGE_TOKEN_INDEX = self.tokenizer.convert_tokens_to_ids("<image>")
        # print("tokenizer idx in skywork_unipic_dev=", self.tokenizer.convert_tokens_to_ids("<image>"))

        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_src.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX]
        )

        # --------------------------------------------------
        # 3. MAE-style 重建
        # --------------------------------------------------

        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m * n, -1)
        orders = self.mar.sample_orders(bsz=b, seq_len=m*n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)
        x_enc = self.forward_mae_encoder(x, mask,
                                         inputs_embeds=inputs_embeds,
                                         attention_mask=attention_mask)
        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n), x_con=x_con)

        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)
        return loss



    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type, batch_data in data_dict.items():
            if 'text2image' in data_type:
                loss = self.text2image_loss(batch_data)
            elif 'image2text' in data_type:
                loss = self.image2text_loss(batch_data)
            elif 'image_edit' in data_type:
                loss = self.image_edit_loss(batch_data)
            else:
                raise NotImplementedError(f"Unknown data_type: {data_type}")
            weight = self.loss_weights.get(data_type, 1.0)
            losses[f'loss_{data_type}'] = loss * weight
        return losses






