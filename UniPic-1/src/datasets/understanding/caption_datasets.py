from torch.utils.data import Dataset
from PIL import Image
import os
import io
import json
import random
import torch
import numpy as np
from einops import rearrange
try:
    from aoss_client.client import Client
except:
    try:
        from petrel_client.client import Client
    except:
        Client = None
from glob import glob
from xtuner.registry import BUILDER
from xtuner.dataset.utils import expand2square
from src.datasets.utils import crop2square, encode_fn
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from src.datasets.understanding.caption_prompts import dense_prompts, short_prompts
from typing import List, Dict, Any, Optional,Callable,Tuple


@BUILDER.register_module()
class CaptionDataset(Dataset):
    def __init__(self,
                 data_path,
                 local_folder,
                 image_size,
                 ceph_folder=None,
                 ceph_config=None,
                 tokenizer=None,
                 template_map_fn=None,
                 max_length=2048,
                 min_image_size=80,
                 image_length=256,
                 pad_image=True,
                 brief=False,
                 cap_folder=None,
                 cap_source='caption',
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.local_folder = local_folder
        self.cap_folder = local_folder if cap_folder is None else cap_folder
        self.cap_source = cap_source

        self.image_size = image_size

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = template_map_fn['template']
        self.template_map_fn = BUILDER.build(template_map_fn)
        self.max_length = max_length
        self.image_length = image_length
        self.pad_image = pad_image
        self.min_image_size = min_image_size

        self.FILE_CLIENT = None
        self.ceph_folder = ceph_folder
        self.ceph_config = ceph_config
        self.use_ceph = ((Client is not None) and (ceph_folder is not None)
                         and (ceph_config is not None) and os.path.exists(ceph_config))

        self.brief = brief
        self.caption_prompts = short_prompts if self.brief else dense_prompts

    def _load_data(self, data_path: str):      # image path and annotation path are saved in a json file
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            json_files = glob(f"{data_path}/*.json")
            data_list = []
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data_list += json.load(f)

            self.data_list = data_list

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_ceph(self, ceph_path):
        if self.FILE_CLIENT is None:
            self.FILE_CLIENT = Client(self.ceph_config)
        data_bytes = self.FILE_CLIENT.get(ceph_path)

        return io.BytesIO(data_bytes)

    def _read_image(self, image_file):
        if self.use_ceph:
            image = Image.open(
                self._read_ceph(
                    os.path.join(self.ceph_folder, image_file)
                )
            )
        else:
            image = Image.open(
                os.path.join(self.local_folder, image_file)
            )
        assert image.width > self.min_image_size and image.height > self.min_image_size, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image.convert('RGB')

    def _read_json(self, annotation_file):
        if self.use_ceph:
            annotation = json.load(
                self._read_ceph(
                    os.path.join(self.ceph_folder, annotation_file)
                )
            )
        else:
            with open(os.path.join(self.local_folder, annotation_file), 'r') as f:
                annotation = json.load(f)

        return annotation

    def _process_image(self, image):
        data = dict()
        if self.pad_image:
            image = expand2square(image, (127, 127, 127))
        else:
            image = crop2square(image)

        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)
        return data

    def _process_text(self, text):
        assert DEFAULT_IMAGE_TOKEN not in text, text
        data_dict = dict(conversation=[{'input': f"{DEFAULT_IMAGE_TOKEN}\n{random.choice(self.caption_prompts)}",
                                        'output': text.strip()}])
        data_dict.update(self.template_map_fn(data_dict))
        data_dict.update(encode_fn(data_dict, self.tokenizer, self.max_length,
                                   self.image_length, True, True))

        assert (torch.tensor(data_dict['input_ids']).long() == IMAGE_TOKEN_INDEX).sum() == self.image_length, \
            "Error in image format"

        data_dict['type'] = 'image2text'
        return data_dict

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            data = self._process_image(image)
            del image
            with open(f"{self.cap_folder}/{data_sample['annotation']}", 'r') as f:
                caption = json.load(f)[self.cap_source]
            data.update(self._process_text(caption))

            data.update(image_dir=self.local_folder, image_file=data_sample['image'])

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{data_sample['image']}: {e}", flush=True)
            return self._retry()


@BUILDER.register_module()
class VqaDataset(Dataset):
    """Generic VQA / multimodal conversation dataset with robust IO & validation."""
    # ---------- 初始化 ----------
    def __init__(
        self,
        data_path: str,
        tokenizer,                      # ← 必填参数，放在最前
        template_map_fn: Callable,      # ← 必填参数，放在最前
        img_prefix: Optional[str] = None,
        image_size: int = 512,
        max_length: int = 2048,
        image_length: int = 1089,
        pad_image: bool = True,
        min_image_size: int = 80,
        image_token_patterns: Tuple[str, ...] = ('<image>', '[image]', '<img>'),
        max_retry: int = 5,
    ):
        super().__init__()

        self.img_prefix = img_prefix.rstrip("/") if img_prefix else None
        self.image_size = image_size
        self.max_length = max_length
        self.image_length = image_length
        self.pad_image = pad_image
        self.min_image_size = min_image_size
        self.image_token_patterns = list(image_token_patterns)
        self.max_retry = max_retry

        # 构建 tokenizer 与模板
        self.tokenizer = BUILDER.build(tokenizer)
        self.template_map_fn = BUILDER.build(template_map_fn) if template_map_fn else None

        # 读取 jsonl / 目录
        self.data_list = self._load_jsonl_list(data_path)
        print(f"Loaded {len(self.data_list)} samples from {data_path}")

    # ---------- 数据加载辅助 ----------
    @staticmethod
    def _load_jsonl_list(path: str) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        if path.endswith(".jsonl"):
            files = [path]
        else:
            files = sorted(glob(os.path.join(path, "**/*.jsonl"), recursive=True))

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        return data

    # ---------- 基本接口 ----------
    def __len__(self) -> int:
        return len(self.data_list)

    # ---------- 图像处理 ----------
    def _get_image_path(self, img_file: str) -> str:
        """保持绝对路径不变，否则加前缀"""
        return img_file if os.path.isabs(img_file) else os.path.join(self.img_prefix, img_file)

    def _read_image(self, img_file: str) -> Image.Image:
        img_path = self._get_image_path(img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Cannot open image: {img_path} ({e})")

        w, h = image.size
        if w < self.min_image_size or h < self.min_image_size:
            raise ValueError(f"Image too small: {img_path} ({w}x{h})")
        ratio = w / h
        if not (0.1 < ratio < 10):
            raise ValueError(f"Odd aspect ratio ({ratio:.3f}) for {img_path}")

        # pad / crop
        image = expand2square(image, (127, 127, 127)) if self.pad_image else crop2square(image)
        image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)

        px = torch.from_numpy(np.asarray(image)).float() / 255.0
        px = 2 * px - 1.0
        px = rearrange(px, "h w c -> c h w")  # CHW
        return px

    # ---------- 对话处理 ----------
    def _replace_image_tokens(self, txt: str) -> str:
        for pat in self.image_token_patterns:
            if pat in txt:
                txt = txt.replace(pat, str(self.image_token_idx))
        return txt

    def _format_conversation(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        将多个 human/gpt 轮次合并为若干 {'input':..., 'output':...} 对。
        遵循：human → gpt 为一对；若缺失 reply，用占位符。
        """
        pairs = []

        for i in range(0, len(turns), 2):  # 每两回合一对，human 和 gpt
            if i + 1 < len(turns):  # 确保 gpt turn 存在
                human_turn = turns[i]
                gpt_turn = turns[i + 1]

                human_content = human_turn.get("value", "").strip()
                gpt_content = gpt_turn.get("value", "").strip()

                if not human_content.lstrip().startswith("<image>"):
                    human_content = f"<image>\n{human_content}"

                if not human_content or not gpt_content:  # 如果某一方没有内容，跳过该对话
                    continue

                # 只在 human turn 中加入图像 token
                # human_content = self._replace_image_tokens(human_content)  # 替换成 image_token_idx

                pairs.append({"input": human_content, "output": gpt_content})

        data_dict = {"conversation": pairs}
        data_dict_ori = data_dict
        if self.template_map_fn:
            data_dict = self.template_map_fn(data_dict)

        # 对输入进行编码
        data_dict = encode_fn(
            data_dict,
            self.tokenizer,
            self.max_length,
            self.image_length,
            input_ids_with_output=True,
            with_image_token=True,
            # 额外把 image_token_idx 传进去
            image_token_idx=self.image_token_idx
        )

        # 动态校验：确保至少出现一次图像 token
        img_tokens = (torch.tensor(data_dict["input_ids"]) == self.image_token_idx).sum().item()

        # 使用f-string优化打印格式，确保输出类型安全
        print(f"[校验日志] input_ids长度: {len(data_dict['input_ids'])}, 图像token出现次数: {img_tokens}\n")
        # print(f"[校验日志] input_ids: {data_dict.get('input_ids', '未设置')}\n")
        if img_tokens != 1088:
            print(f"[异常对话]:{data_dict_ori}")

        data_dict["type"] = "image2text"  # 设置数据类型为 image2text
        return data_dict


    # ---------- 主接口 ----------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for attempt in range(self.max_retry):
            try:
                sample = self.data_list[idx]
                img_tensor = self._read_image(sample["image"])
                text_data = self._format_conversation(sample.get("conversations", []))
                return {
                    **text_data,
                    "pixel_values": img_tensor,
                    "image_file": sample["image"],
                }
            except Exception as e:
                print(f"[Retry {attempt+1}/{self.max_retry}] idx={idx} error: {e}")
                idx = random.randint(0, len(self) - 1)

        # 若多次失败则抛异常
        raise RuntimeError(f"Failed to fetch valid sample after {self.max_retry} retries.")