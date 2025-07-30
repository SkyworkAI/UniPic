import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from src.builder import BUILDER
from einops import rearrange
import argparse


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--image", type=str, default="data/sample.png")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    if args.checkpoint is not None:
        print(f"Load checkpoint: {args.checkpoint}", flush=True)
        checkpoint = torch.load(args.checkpoint)
        info = model.load_state_dict(checkpoint, strict=False)

    special_tokens_dict = {'additional_special_tokens': ["<image>", ]}
    num_added_toks = model.tokenizer.add_special_tokens(special_tokens_dict)
    # assert num_added_toks == 1

    image_token_idx = model.tokenizer.encode("<image>", add_special_tokens=False)[-1]
    print(f"Image token: {model.tokenizer.decode(image_token_idx)}")

    image = Image.open(args.image).convert('RGB')

    image = expand2square(
        image, (127, 127, 127))
    image = image.resize(size=(args.image_size, args.image_size))
    image = torch.from_numpy(np.array(image)).to(dtype=model.dtype, device=model.device)
    image = rearrange(image, 'h w c -> c h w')[None]
    image = 2 * (image / 255) - 1

    prompt = model.prompt_template['INSTRUCTION'].format(input="<image>\n" + args.prompt)
    assert '<image>' in prompt
    image_length = (args.image_size // 16) ** 2 + 64
    prompt = prompt.replace('<image>', '<image>'*image_length)
    input_ids = model.tokenizer.encode(
        prompt, add_special_tokens=True, return_tensors='pt').cuda()
    with torch.no_grad():
        _, z_enc = model.extract_visual_feature(model.encode(image))
    inputs_embeds = z_enc.new_zeros(*input_ids.shape, model.llm.config.hidden_size)
    inputs_embeds[input_ids == image_token_idx] = z_enc.flatten(0, 1)
    inputs_embeds[input_ids != image_token_idx] = model.llm.get_input_embeddings()(
        input_ids[input_ids != image_token_idx]
    )
    with torch.no_grad():
        output = model.llm.generate(inputs_embeds=inputs_embeds,
                                    use_cache=True,
                                    do_sample=False,
                                    max_new_tokens=1024,
                                    eos_token_id=model.tokenizer.eos_token_id,
                                    pad_token_id=model.tokenizer.pad_token_id
                                    if model.tokenizer.pad_token_id is not None else
                                    model.tokenizer.eos_token_id
                                    )
    print(model.tokenizer.decode(output[0]))
