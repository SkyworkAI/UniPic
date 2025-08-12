import torch
from PIL import Image
from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
from unipicv2.stable_diffusion_3_conditioner import StableDiffusion3Conditioner
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL


pretrained_model_name_or_path = "/path/to/unipicv2_qwen2_5_vl_7b_sd_3_5m_kontext"

transformer = SD3Transformer2DKontextModel.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16).cuda()

vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    subfolder="vae",
    torch_dtype=torch.bfloat16).cuda()

lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", ).cuda()
processor = Qwen2_5_VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor.chat_template = processor.chat_template.replace(
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
    "")


conditioner = StableDiffusion3Conditioner.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    subfolder="conditioner",
    torch_dtype=torch.bfloat16
).cuda()


scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")


pipeline = StableDiffusion3KontextPipeline(
    transformer=transformer,
    vae=vae,
    text_encoder=None,
    tokenizer=None,
    text_encoder_2=None,
    tokenizer_2=None,
    text_encoder_3=None,
    tokenizer_3=None,
    scheduler=scheduler)


### text-to-image generation
prompt = 'a pig with wings and a top hat flying over a happy futuristic scifi city'
negative_prompt = ''

# Sample messages for batch inference
messages = [[{"role": "user", "content": [{"type": "text", "text": f'Generate an image: {txt}'}, ],}, ]
            for txt in [prompt, negative_prompt]]

texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
inputs = processor(
    text=texts,
    images=None,
    videos=None,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
input_ids = torch.cat([input_ids, input_ids.new_zeros(2, conditioner.config.num_queries)], dim=1)
attention_mask = torch.cat([attention_mask, attention_mask.new_ones(2, conditioner.config.num_queries)], dim=1)
inputs_embeds = lmm.get_input_embeddings()(input_ids)
inputs_embeds[:, -conditioner.config.num_queries:] = conditioner.meta_queries[None].expand(2, -1, -1)

outputs = lmm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
hidden_states = outputs.last_hidden_state[:, -conditioner.config.num_queries:]
prompt_embeds, pooled_prompt_embeds = conditioner(hidden_states)

image = pipeline(
    prompt_embeds=prompt_embeds[:1],
    pooled_prompt_embeds=pooled_prompt_embeds[:1],
    negative_prompt_embeds=prompt_embeds[1:],
    negative_pooled_prompt_embeds=pooled_prompt_embeds[1:],
    height=512,
    width=384,
    num_inference_steps=50,
    guidance_scale=3.5,
    generator=torch.Generator(device=transformer.device).manual_seed(42)
).images[0]

image.save("text2image.png")



### image editing

def fix_longer_edge(x, image_size, factor=32):
    w, h = x.size
    if w >= h:
        target_w = image_size
        target_h = h * (target_w / w)
        target_h = round(target_h / factor) * factor
    else:
        target_h = image_size
        target_w = w * (target_h / h)
        target_w = round(target_w / factor) * factor

    x = x.resize(size=(target_w, target_h))

    return x


image = Image.open("text2image.png")
image = fix_longer_edge(image, image_size=512)


prompt = "remove the pig's hat"
negative_prompt = "blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas."

# Sample messages for batch inference
messages = [[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": txt}, ],},]
            for txt in [prompt, negative_prompt]]

texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]

min_pixels = max_pixels = int(image.height * 28 / 32 * image.width * 28 / 32)

inputs = processor(
    text=texts,
    images=[image]*2,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    videos=None,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

input_ids, attention_mask, pixel_values, image_grid_thw = \
    inputs.input_ids, inputs.attention_mask, inputs.pixel_values, inputs.image_grid_thw
input_ids = torch.cat([input_ids, input_ids.new_zeros(2, conditioner.config.num_queries)], dim=1)
attention_mask = torch.cat([attention_mask, attention_mask.new_ones(2, conditioner.config.num_queries)], dim=1)
inputs_embeds = lmm.get_input_embeddings()(input_ids)
inputs_embeds[:, -conditioner.config.num_queries:] = conditioner.meta_queries[None].expand(2, -1, -1)
image_embeds = lmm.visual(pixel_values, grid_thw=image_grid_thw)
image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')

inputs_embeds[input_ids == image_token_id] = image_embeds

lmm.model.rope_deltas = None
outputs = lmm.model(inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    image_grid_thw=image_grid_thw,
                    use_cache=False)

hidden_states = outputs.last_hidden_state[:, -conditioner.config.num_queries:]
prompt_embeds, pooled_prompt_embeds = conditioner(hidden_states)

image = pipeline(
    image=image,
    negative_prompt=None,
    prompt_embeds=prompt_embeds[:1],
    pooled_prompt_embeds=pooled_prompt_embeds[:1],
    negative_prompt_embeds=prompt_embeds[1:],
    negative_pooled_prompt_embeds=pooled_prompt_embeds[1:],
    height=image.height,
    width=image.width,
    num_inference_steps=50,
    guidance_scale=3.5,
    generator=torch.Generator(device=transformer.device).manual_seed(42)
).images[0]

image.save("image_editing.png")
