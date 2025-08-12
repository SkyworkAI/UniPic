import torch
from PIL import Image
from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel


from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import (CLIPTextModelWithProjection, CLIPTokenizer,
                          T5EncoderModel, T5TokenizerFast)

pretrained_model_name_or_path = "/path/to/unipicv2_sd_3_5m_kontext"

transformer = SD3Transformer2DKontextModel.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    subfolder="transformer",
    torch_dtype=torch.bfloat16).cuda()

vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    subfolder="vae",
    torch_dtype=torch.bfloat16).cuda()

text_encoder = CLIPTextModelWithProjection.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder",
    torch_dtype=torch.bfloat16).cuda()
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16).cuda()

tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
text_encoder_3 = T5EncoderModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="text_encoder_3",
    torch_dtype=torch.bfloat16).cuda()

tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")


pipeline = StableDiffusion3KontextPipeline(
    transformer=transformer,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    text_encoder_3=text_encoder_3,
    tokenizer_3=tokenizer_3,
    scheduler=scheduler)


### text-to-image generation

image = pipeline(
    prompt='a pig with wings and a top hat flying over a happy futuristic scifi city',
    negative_prompt='',
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


negative_prompt = "blurry, low quality, low resolution, distorted, deformed, broken content, missing parts, damaged details, artifacts, glitch, noise, pixelated, grainy, compression artifacts, bad composition, wrong proportion, incomplete editing, unfinished, unedited areas."

image = pipeline(
    image=image,
    prompt="remove the pig's hat",
    negative_prompt=negative_prompt,
    height=image.height,
    width=image.width,
    num_inference_steps=50,
    guidance_scale=3.5,
    generator=torch.Generator(device=transformer.device).manual_seed(42)
).images[0]
image.save("image_editing.png")
