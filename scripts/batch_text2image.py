import json
import os
import copy
import torch
import argparse
from tqdm import tqdm
from xtuner.registry import BUILDER
from mmengine.config import Config
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from einops import rearrange


class JsonDataset(Dataset):
    def __init__(self, data_path):

        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])
        data_dict['sample_id'] = idx

        return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--data', default='path/to/xxx.json', type=str)
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument("--cfg_prompt", type=str, default='Generate an image.')
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--num_iter', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--grid_size', type=int, default=2)
    args = parser.parse_args()

    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)

    config = Config.fromfile(args.config)

    print(f'Device: {accelerator.device}', flush=True)

    dataset = JsonDataset(data_path=args.data)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=lambda x: x
                            )

    model = BUILDER.build(config.model)
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=accelerator.device)
    model = model.to(model.dtype)
    model.eval()

    dataloader = accelerator.prepare(dataloader)

    print(f'Number of samples: {len(dataloader)}', flush=True)
    m = n = args.image_size // 16

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)

    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        device_idx = accelerator.process_index

        prompts = [
            model.prompt_template['INSTRUCTION'].format(
                input=f"Generate an image: {data_sample['prompt'].strip()}.")
            for data_sample in data_samples
        ] * (args.grid_size ** 2)

        if args.cfg != 1.0:
            prompts += [model.prompt_template['INSTRUCTION'].format(input=args.cfg_prompt)] * (4 * len(data_samples))

        inputs = model.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True).to(accelerator.device)

        images = model.sample(**inputs, num_iter=args.num_iter, cfg=args.cfg, cfg_schedule=args.cfg_schedule,
                              temperature=args.temperature, progress=False, image_shape=(m, n))
        images = rearrange(images, '(m n b) c h w -> b (m h) (n w) c', m=args.grid_size, n=args.grid_size)

        images = torch.clamp(
            127.5 * images + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for image, data_sample in zip(images, data_samples):
            sample_id = data_sample['sample_id']
            with open(f"{args.output}/{sample_id:08d}.json", "w") as f:
                json.dump(obj=data_sample, fp=f)
            Image.fromarray(image).save(f"{args.output}/{sample_id:08d}.jpg")
