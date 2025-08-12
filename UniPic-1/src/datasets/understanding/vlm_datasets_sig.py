from torch.utils.data import Dataset
from PIL import Image
import os
import json
import random
import torch
import numpy as np
from einops import rearrange
from xtuner.registry import BUILDER
from xtuner.dataset.utils import expand2square
from src.datasets.utils import crop2square, encode_fn, load_jsonl
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from transformers import AutoImageProcessor


class VLMDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_size,
        tokenizer=None,
        template_map_fn=None,
        max_length=2048,
        min_image_size=80,
        pad_image=True,
        local_folder="",
        key_value="conversations",
    ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.image_size = image_size

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = template_map_fn["template"]
        self.template_map_fn = BUILDER.build(template_map_fn)
        self.max_length = max_length
        self.pad_image = pad_image
        self.min_image_size = min_image_size
        self.key_value = key_value
        self.processor = AutoImageProcessor.from_pretrained(
            "checkpoint/siglip2-so400m-patch16-512"
        )
        self.metainfo = {'task' :'unified'}
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        m = n = self.image_size // 16
        self.image_token_repeat = m * n + 64

        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.image_token_idx = self.tokenizer.convert_tokens_to_ids("<image>")
        print(f"Registered <image> token at index {self.image_token_idx}")

    def _load_data(
        self, data_path: str
    ):  # image path and annotation path are saved in a json file
        self.data_list = load_jsonl(data_path)
        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def full_init(self):
        """Dummy full_init to be compatible with MMEngine ConcatDataset."""
        return
    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        image = Image.open(image_file)
        assert (
            image.width > self.min_image_size and image.height > self.min_image_size
        ), f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image.convert("RGB")

    # def _process_image(self, image):
    #     data = dict()
    #     # if self.pad_image:
    #     #     image = expand2square(image, (127, 127, 127))
    #     # else:
    #     #     image = crop2square(image)

    #     # image = image.resize(size=(self.image_size, self.image_size))
    #     # pixel_values = torch.from_numpy(np.array(image)).float()
    #     # pixel_values = pixel_values / 255
    #     # pixel_values = 2 * pixel_values - 1
    #     # pixel_values = rearrange(pixel_values, "h w c -> c h w")
    #     image = image.resize((self.image_size, self.image_size))
    #     inputs = self.processor(images=image, return_tensors="pt")
    #     pixel_values = inputs["pixel_values"].squeeze(0)

    #     data.update(pixel_values=pixel_values)
    #     return data


    def _process_image(self, image: Image.Image):
        # 1) 可选 crop/pad to square
        if self.pad_image:
            image = crop2square(image)
        # 2) 手动 resize 到指定大小
        image = image.resize((self.image_size, self.image_size))
        # 3) to tensor & normalize
        arr = np.array(image).astype(np.float32) / 255.0          # HWC
        arr = 2 * arr - 1                                         # [-1,1]
        tensor = torch.from_numpy(arr)                            # HWC
        tensor = rearrange(tensor, "h w c -> c h w")             # CHW
        return {"pixel_values": tensor}
    def _process_text(self, question, answer):
        data_dict = dict(
            conversation=[
                {
                    "input": f"{self.DEFAULT_IMAGE_TOKEN}\n{question}",
                    "output": answer,
                }
            ]
        )
        data_dict.update(self.template_map_fn(data_dict))
        data_dict.update(
            encode_fn(
                example=data_dict,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                image_length=self.image_token_repeat,
                input_ids_with_output=True,
                with_image_token=True,
                truncation='right',
                image_token_idx=self.image_token_idx,
                image_token_str=self.DEFAULT_IMAGE_TOKEN,
            )
        )

        # assert (
        #     torch.tensor(data_dict["input_ids"]).long() == self.image_token_idx
        # ).sum() == self.image_length, "Error in image format"

        data_dict["type"] = "image2text"
        return data_dict

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample["image"]).convert("RGB")
            data = self._process_image(image)
            del image
            question = (
                data_sample[self.key_value][0]["value"]
                .replace("<image>", "")
                .strip()
            )
            answer = (
                data_sample[self.key_value][1]["value"]
                .replace("<image>", "")
                .strip()
            )

            data.update(self._process_text(question, answer))

            data.update(image_file=data_sample["image"])

            return data

        except Exception as e:
            print(
                f"Error when reading data_sample:{data_sample},{self.data_path}:{data_sample['image']}: {e}",
                flush=True,
            )
            return self._retry()
