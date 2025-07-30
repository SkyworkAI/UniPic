from torch.utils.data import Dataset
from PIL import Image
import os
import json
import random
import torch
import numpy as np
from einops import rearrange
from xtuner.registry import BUILDER
from mmengine.registry import DATASETS
from src.datasets.utils import crop2square
from glob import glob
from typing import List, Dict, Any, Optional
import mmap
import struct
from src.datasets.utils import crop2square, encode_fn
from xtuner.utils import DEFAULT_IMAGE_TOKEN


@BUILDER.register_module()
class Text2ImageDataset(Dataset):
    def __init__(self,
                 data_path,
                 local_folder,
                 image_size,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=True,
                 cap_source='caption',
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.unconditional = unconditional
        self.local_folder = local_folder
        self.cap_source = cap_source
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)

        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.metainfo = {'task': 'unified'}
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)



    def _load_data(self, data_path):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)
    
    def full_init(self):
        """Dummy full_init to be compatible with MMEngine ConcatDataset."""
        return
    

    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        image = Image.open(os.path.join(self.local_folder, image_file))
        assert image.width > 8 and image.height > 8, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image

    def _process_text(self, text):
        if random.uniform(0, 1) < self.unconditional:
            prompt = "Generate an image."
        else:
            prompt = f"Generate an image: {text.strip()}"
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')[0]

        return dict(input_ids=input_ids[:self.max_length])

    def _process_image(self, image):
        data = dict()

        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))

        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)

        return data

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')

            caption = data_sample[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()

@DATASETS.register_module()
@BUILDER.register_module()
class LargeText2ImageDataset(Text2ImageDataset):
    # self.data_list only contains paths of images and captions

    def __init__(self, cap_folder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_folder = self.local_folder if cap_folder is None else cap_folder

    def _load_data(self, data_path):      # image path and annotation path are saved in a json file
        if data_path.endswith(".json"):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            self.data_list = []
            json_files = glob(f'{data_path}/*.json')
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    self.data_list += json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            with open(f"{self.cap_folder}/{data_sample['annotation']}", 'r') as f:
                caption = json.load(f)[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{data_sample}: {e}", flush=True)
            return self._retry()


@DATASETS.register_module()
@BUILDER.register_module()
class MMapT2IDataset(Dataset):
    """
    Map-style Text2Image Dataset with mmap-based random access.
    一次性在 __init__ 打开 mmap；__getitem__ O(1) 读取指定行。
    """
    def __init__(
        self,
        jsonl_path: str,
        idx_path: str,
        image_size: int,
        tokenizer: Optional[Dict] = None,
        template_map_fn: Optional[Dict] = None,
        cap_source: str = "prompt",
        max_length: int = 2048,
        image_length: int = 512,
        unconditional: float = 0.01,
        crop_image: bool = False,
    ):
        super().__init__()

        # ---------- 基础参数 ----------
        self.jsonl_path = jsonl_path
        self.image_size = image_size
        self.cap_source = cap_source
        self.max_length = max_length
        self.unconditional = unconditional
        self.crop_image = crop_image

        # ---------- tokenizer / template ----------
        self.tokenizer = BUILDER.build(tokenizer)
        self.template_map_fn = template_map_fn

        # ---------- mmap 加载 ----------
        self._open_mmap(jsonl_path, idx_path)
        self.metainfo = {'task' :'unified'}
    # ===== mmap & index =====
    def _open_mmap(self, jsonl_path: str, idx_path: str):
        # mmap 文件
        self._jsonl_fp = open(jsonl_path, "r+b")
        self._mm = mmap.mmap(self._jsonl_fp.fileno(), 0, access=mmap.ACCESS_READ)

        # 读取 offset 索引
        with open(idx_path, "rb") as f:
            nlines = struct.unpack("<Q", f.read(8))[0]
            self._offsets = np.frombuffer(f.read(8 * nlines), dtype=np.uint64)
        print(f"[MMapT2IDataset] {jsonl_path}: {nlines} lines indexed")

    def __len__(self) -> int:
        return self._offsets.size

    def full_init(self):
        """Dummy full_init to be compatible with MMEngine ConcatDataset."""
        return
    def _read_line(self, idx: int) -> str:
        off = int(self._offsets[idx])
        self._mm.seek(off)
        return self._mm.readline().decode("utf-8")

    # ===== 核心处理 =====
    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")

        # 预处理：裁剪成方形 / pad
        if self.crop_image:
            img = crop2square(img)
        else:
            target_size = max(img.size)
            img = img.resize((target_size, target_size))

        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.uint8)          # HWC uint8
        px = torch.as_tensor(arr).float() / 255.0      # 0-1
        px = 2 * px - 1                                # -1 ~ 1
        return rearrange(px, "h w c -> c h w")         # CHW

    def _build_prompt(self, caption: str) -> torch.Tensor:
        if random.random() < self.unconditional:
            caption = "Generate an image."
        else:
            caption = f"Generate an image: {caption.strip()}"

        instr = self.template_map_fn["INSTRUCTION"].format(input=caption)
        ids = self.tokenizer.encode(
            instr, add_special_tokens=True, return_tensors="pt"
        )[0][: self.max_length]
        return ids

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1) 取 jsonl 行
        sample = json.loads(self._read_line(idx))

        # 2) 加载 & 处理图像
        pixel_values = self._load_image(sample["image"])

        # 3) 处理文本
        caption = sample.get(self.cap_source, "")
        input_ids = self._build_prompt(caption)

        # 4) 打包
        data = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            type="text2image",
            image_file=sample["image"],
            idx=idx,
        )
        return data


@DATASETS.register_module()
@BUILDER.register_module()
class ReconstructDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 image_size: int,
                 tokenizer=None,
                 prompt_template=None,
                 cap_source: str = "prompt",
                 max_length: int = 8192,
                 crop_image: bool = True,
                 img_prefix: str = ""):
        super().__init__()
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.prompt_template = prompt_template
        self.cap_source = cap_source
        self.max_length = max_length
        self.crop_image = crop_image
        self.img_prefix = img_prefix
        self._load_data(data_path)

        m = n = self.image_size // 16
        self.image_token_repeat = m * n + 64
        self.metainfo = {'task': 'unified'}

    def full_init(self):
        """Dummy full_init to be compatible with MMEngine ConcatDataset."""
        return

    def _load_data(self, path):
        with open(path) as f:
            self.data_list = [json.loads(l) for l in f]
        print(f"[I2ICaptionReconstructDataset] Loaded {len(self.data_list)} samples from {path}")

    def _add_prefix(self, rel):
        return os.path.join(self.img_prefix, rel.lstrip("/")) if self.img_prefix else rel

    def _read_image(self, path):
        img = Image.open(path).convert("RGB")
        assert img.width > 8 and img.height > 8 and 0.1 < img.width / img.height < 10
        return img

    # ---------- preprocess ----------
    def _process_image(self, img):
        img = crop2square(img) if self.crop_image else img.resize((max(img.size),)*2)
        img = img.resize((self.image_size, self.image_size))
        px  = torch.from_numpy(np.array(img)).float() / 255.
        px  = 2 * px - 1
        return rearrange(px, "h w c -> c h w")

    def _encode_prompt(self, text):
        # for bad_token in ["[IMAGE]", "<image_placeholder>", "<image_plaeholder>"]:
        #     text = text.replace(bad_token, "")
        text = "Repeat this image."
        prompt_in = f"<image>\n{text.strip()}"
        prompt = self.prompt_template["INSTRUCTION"].format(input=prompt_in)
        prompt = prompt.replace("<image>", "<image>" * self.image_token_repeat)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")[0]
        mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids[:self.max_length], mask[:self.max_length]

    def __len__(self):
        return len(self.data_list)

    def _retry(self):
        return self.__getitem__(random.randrange(len(self)))

    def __getitem__(self, idx):
        try:
            sample = self.data_list[idx]
            src_img = self._read_image(self._add_prefix(sample["image"]))
            tgt_img = src_img
            caption = sample[self.cap_source]

            px_src = self._process_image(src_img)
            px_tgt = self._process_image(tgt_img)
            input_ids, mask = self._encode_prompt(caption)

            return {
                "pixel_values_src": px_src,
                "pixel_values": px_tgt,
                "input_ids": input_ids,
                "attention_mask": mask,
                "type": "image_edit"
            }
        except Exception as e:
            print(f"[I2ICaptionReconstructDataset] Error @ {idx}: {e}")
            return self._retry()

@DATASETS.register_module()
@BUILDER.register_module()
class UncondReconstructDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 image_size: int,
                 tokenizer=None,
                 prompt_template=None,
                 cap_source: str = "prompt",
                 max_length: int = 8192,
                 crop_image: bool = True,
                 img_prefix: str = ""):
        super().__init__()
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image
        self.img_prefix = img_prefix
        self.cap_source = cap_source


        self._load_data(data_path)

        # 计算 image token 展开数量
        m = n = self.image_size // 16
        self.image_token_repeat = m * n + 64
        self.metainfo = {'task': 'unified'}
        
    def _load_data(self, path):
        with open(path) as f:
            self.data_list = [json.loads(l) for l in f]
        print(f"[I2IUncondReconstructDataset] Loaded {len(self.data_list)} samples from {path}")

    def _add_prefix(self, rel_path):
        return os.path.join(self.img_prefix, rel_path.lstrip("/")) if self.img_prefix else rel_path

    def full_init(self):
        """Dummy full_init to be compatible with MMEngine ConcatDataset."""
        return
    def _read_image(self, path):
        image = Image.open(path).convert("RGB")
        assert image.width > 8 and image.height > 8 and 0.1 < image.width / image.height < 10
        return image


    # ---------- preprocess ----------
    def _process_image(self, img):
        img = crop2square(img) if self.crop_image else img.resize((max(img.size),)*2)
        img = img.resize((self.image_size, self.image_size))
        px  = torch.from_numpy(np.array(img)).float() / 255.
        px  = 2 * px - 1
        return rearrange(px, "h w c -> c h w")

    def __len__(self):
        return len(self.data_list)

    def _retry(self, max_tries=5):
        for _ in range(max_tries):
            try:
                return self.__getitem__(random.randrange(len(self)))
            except Exception:
                continue
        raise RuntimeError("Exceeded max retries in I2IUncondReconstructDataset")

    def __getitem__(self, idx):
        try:
            sample = self.data_list[idx]
            path = self._add_prefix(sample["image"])
            img = self._read_image(path)
            px = self._process_image(img)

            # ==== 填入空文本 ====
            input_ids = torch.zeros(0, dtype=torch.long)
            attention_mask = torch.zeros(0, dtype=torch.long)

            return {
                "pixel_values_src": px,
                "pixel_values": px.clone(),
                "type": "image_edit",
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                # 重建任务不再输出 input_ids / attention_mask
            }
        except Exception as e:
            print(f"[I2IUncondReconstructDataset] Error @ {idx}: {e}")
            return self._retry()



@DATASETS.register_module()
@BUILDER.register_module()
class Text2ImageJSONLDataset(Dataset):
    def __init__(self,
                 data_path,
                 image_size,
                 tokenizer=None,
                 prompt_template=None,
                 cap_source='prompt',
                 max_length=1024,
                 unconditional=0.1,
                 crop_image=True,
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.prompt_template = prompt_template
        self.cap_source = cap_source
        self.max_length = max_length
        self.unconditional = unconditional
        self.crop_image = crop_image
        self.metainfo = {'task': 'unified'}

    def _load_data(self, data_path):
        self.data_list = []
        with open(data_path, 'r') as f:
            for line in f:
                self.data_list.append(json.loads(line.strip()))
        print(f"Loaded {len(self.data_list)} samples from {data_path}")

    def full_init(self):
        """Dummy full_init for MMEngine ConcatDataset compatibility."""
        pass
    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        image = Image.open(image_file).convert('RGB')
        assert image.width > 8 and image.height > 8
        assert 0.1 < image.width / image.height < 10
        return image

    def _process_image(self, image):
        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize((target_size, target_size))

        image = image.resize((self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float() / 255.0
        pixel_values = 2 * pixel_values - 1  # [-1, 1]
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return dict(pixel_values=pixel_values)

    def _process_text(self, text):
        if random.uniform(0, 1) < self.unconditional:
            text = "Generate an image."
        else:
            text = f"Generate an image: {text.strip()}"
        prompt = self.prompt_template['INSTRUCTION'].format(input=text)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')[0]
        return dict(input_ids=input_ids[:self.max_length])

    def _retry(self):
        return self.__getitem__(random.randint(0, len(self.data_list) - 1))

    def __getitem__(self, idx):
        try:
            sample = self.data_list[idx]
            image = self._read_image(sample['image'])
            caption = sample[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data
        except Exception as e:
            print(f"[JSONLDataset] Error reading sample #{idx}: {e}")
            return self._retry()



# 纯文生图没有占位符的问题，下面编辑数据集需要考虑占位符
@DATASETS.register_module()
@BUILDER.register_module()
class ImageEditJSONLDataset(Dataset):
    """
    Dataset for <src, tgt, prompt> image editing, now decoupled from tokenization logic.
    """
    def __init__(self,
                 data_path: str,
                 image_size: int,
                 tokenizer=None,
                 prompt_template=None,
                 max_length: int = 8192,
                 cap_source: str = "prompt",
                 unconditional: float = 0,
                 crop_image: bool = False,
                 img_prefix: str = ""):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.cap_source = cap_source
        self.unconditional = unconditional
        self.crop_image = crop_image
        self.img_prefix = img_prefix
        self._load_data(data_path)
        # Calculate image token repetition length, consistent with inference.
        m = n = self.image_size // 16
        self.image_token_repeat = m * n + 64
        self.metainfo = {'task': 'unified'}

        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.image_token_idx = self.tokenizer.convert_tokens_to_ids("<image>")
        print(f"Registered <image> token at index {self.image_token_idx}")

    def _load_data(self, path):
        with open(path) as f:
            self.data_list = [json.loads(l) for l in f]
        print(f"[ImageEditJSONLDataset] Loaded {len(self.data_list)} samples from {path}")

    def full_init(self):
        """Dummy full_init for MMEngine ConcatDataset compatibility."""
        pass

    def _add_prefix(self, rel_path):
        return os.path.join(self.img_prefix, rel_path.lstrip("/")) if self.img_prefix else rel_path

    def _read_image(self, path):
        path = path.replace("datasets_vlm02", "datasets_vlm")
        img = Image.open(path).convert("RGB")
        assert img.width > 8 and img.height > 8 and 0.1 < img.width / img.height < 10
        return img

    def _process_image(self, img):
        img = crop2square(img) if self.crop_image else img.resize((max(img.size),) * 2)
        img = img.resize((self.image_size, self.image_size))
        px = torch.from_numpy(np.array(img)).float() / 255.
        px = 2 * px - 1
        return rearrange(px, "h w c -> c h w")

    # --- REFACTORED: This method now only prepares the raw prompt text ---
    def _prepare_prompt_text(self, raw_text: str):
        """Cleans text and handles unconditional generation."""

        for bad_token in ["[IMAGE]", "<image_placeholder>", "<image_plaeholder>", "<image>"]:
            txt = raw_text.replace(bad_token, "")
        txt = txt.strip()

        if random.random() < self.unconditional:
            txt = "Edit this image."
        return txt

    def _retry(self):
        return self.__getitem__(random.randrange(len(self)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            sample = self.data_list[idx]
            src_path, tgt_path = map(self._add_prefix, [sample["images"][0], sample["image"]])
            src_img, tgt_img = map(self._read_image, [src_path, tgt_path])

            px_src, px_tgt = map(self._process_image, [src_img, tgt_img])
            
            # --- MODIFIED: Call the unified encode_fn ---
            # 1. Prepare the raw prompt string
            prompt_text = self._prepare_prompt_text(sample[self.cap_source])

            # 2. Delegate all encoding and formatting to encode_fn
            encoded_text = encode_fn(
                example=prompt_text,
                tokenizer=self.tokenizer,
                prompt_template=self.prompt_template,
                max_length=self.max_length,
                image_length=self.image_token_repeat,
                image_token_idx=self.image_token_idx
            )

            return {
                    "pixel_values_src": px_src,
                    "pixel_values": px_tgt,
                    "input_ids": torch.tensor(encoded_text["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded_text["attention_mask"], dtype=torch.long),
                    "type": "image_edit",
                }
        except Exception as e:
            print(f"[ImageEditJSONLDataset] Error @ {idx}: {e} from {self.data_path}")
            return self._retry()



