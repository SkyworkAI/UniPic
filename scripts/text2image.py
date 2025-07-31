import torch
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from einops import rearrange


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default='A glossy-coated golden retriever stands on the park lawn beside a life-sized penguin statue.')
    parser.add_argument("--cfg_prompt", type=str, default='Generate an image.')
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--num_iter', type=int, default=32)
    parser.add_argument('--grid_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--output', type=str, default='output.jpg')
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    checkpoint = torch.load(args.checkpoint)
    info = model.load_state_dict(checkpoint, strict=False)
    args.prompt = f"Generate an image: {args.prompt}"
    print(args.prompt, flush=True)
    class_info = model.prepare_text_conditions(args.prompt, args.cfg_prompt)

    input_ids = class_info['input_ids']
    attention_mask = class_info['attention_mask']

    assert len(input_ids) == 2    # the last one is unconditional prompt
    if args.cfg == 1.0:
        input_ids = input_ids[:1]
        attention_mask = attention_mask[:1]

    # repeat
    bsz = args.grid_size ** 2
    if args.cfg != 1.0:
        input_ids = torch.cat([
            input_ids[:1].expand(bsz, -1),
            input_ids[1:].expand(bsz, -1),
        ])
        attention_mask = torch.cat([
            attention_mask[:1].expand(bsz, -1),
            attention_mask[1:].expand(bsz, -1),
        ])
    else:
        input_ids = input_ids.expand(bsz, -1)
        attention_mask = attention_mask.expand(bsz, -1)
 
    m = n = args.image_size // 16

    samples = model.sample(input_ids=input_ids, attention_mask=attention_mask,
                           num_iter=args.num_iter, cfg=args.cfg, cfg_schedule=args.cfg_schedule,
                           temperature=args.temperature, progress=True, image_shape=(m, n))
    samples = rearrange(samples, '(m n) c h w -> (m h) (n w) c', m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

    Image.fromarray(samples).save(args.output)
