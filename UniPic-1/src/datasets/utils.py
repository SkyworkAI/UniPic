import copy
import random
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
import json


# def crop2square(pil_img):
#     width, height = pil_img.width, pil_img.height

#     if width > height:
#         y0, y1 = 0, height
#         x0 = random.randint(0, width - height)    # [0, w - h]
#         x1 = x0 + height    # [h, w]
#     else:
#         x0, x1 = 0, width
#         y0 = random.randint(0, height - width)   # [0, h - w]
#         y1 = y0 + width     # [w, h]

#     return pil_img.crop(box=(x0, y0, x1, y1))

def crop2square(pil_img):
    width, height = pil_img.width, pil_img.height
    short = min(width, height)
    left = (width - short) // 2
    upper = (height - short) // 2
    return pil_img.crop((left, upper, left + short, upper + short))
def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def encode_fn_original(example,
              tokenizer,
              max_length=None,
              image_length=1,
              input_ids_with_output=True,
              with_image_token=False,
              truncation='right',
              image_token_idx=None,
              image_token_str="<image>"):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    if image_token_idx is None:       # 如果没传，就退回库常量
        image_token_idx = tokenizer.convert_tokens_to_ids("<image>")

    is_multi_turn_conversation = len(example['conversation']) > 1
    if is_multi_turn_conversation:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        if image_token_str in input and with_image_token:
            chunk_encode = [
                tokenizer.encode(chunk, add_special_tokens=False)
                for chunk in input.split(image_token_str)
            ]
            assert len(chunk_encode) == 2
            input_encode = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                input_encode.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    # input_encode.append(IMAGE_TOKEN_INDEX)
                    input_encode += [image_token_idx] * image_length

        else:
            input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        if input_ids_with_output and 'output' in single_turn_conversation:
            # Add output
            output_with_loss = single_turn_conversation.get(
                'output_with_loss', True)
            output = single_turn_conversation['output']

            if image_token_str in output and with_image_token:
                chunk_encode = [
                    tokenizer.encode(chunk, add_special_tokens=False)
                    for chunk in output.split(image_token_str)
                ]
                assert len(chunk_encode) == 2
                output_encode = []
                for idx, cur_chunk_encode in enumerate(chunk_encode):
                    output_encode.extend(cur_chunk_encode)
                    if idx != len(chunk_encode) - 1:
                        output_encode += [image_token_idx] * image_length
            else:
                output_encode = tokenizer.encode(output, add_special_tokens=False)
            # output_encode = tokenizer.encode(output, add_special_tokens=False)
            input_ids += output_encode
            if output_with_loss:
                labels += copy.deepcopy(output_encode)
            else:
                labels += [IGNORE_INDEX] * len(output_encode)
            # Add EOS_TOKEN (with loss)
            if single_turn_conversation.get('need_eos_token', True):
                next_needs_bos_token = True
                input_ids += eos_token_id
                if output_with_loss:
                    labels += copy.deepcopy(eos_token_id)
                else:
                    labels += [IGNORE_INDEX] * len(eos_token_id)
            else:
                next_needs_bos_token = False
            # Add SEP (without loss)
            sep = single_turn_conversation.get('sep', '')
            if sep != '':
                sep_encode = tokenizer.encode(sep, add_special_tokens=False)
                input_ids += sep_encode
                labels += [IGNORE_INDEX] * len(sep_encode)

    if max_length is not None and len(input_ids) > max_length:
        if truncation == 'right':
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        elif truncation == 'left':
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        else:
            assert truncation is None
    return {'input_ids': input_ids, 'labels': labels}



def encode_fn(
    example,
    tokenizer,
    prompt_template=None,
    max_length=None,
    image_length=1,
    input_ids_with_output=True,
    with_image_token=True,
    truncation='right',
    image_token_idx=None,
    image_token_str="<image>",
):
    """
    A versatile encoding function for both image-to-text (conversation) and text-to-image/image-editing tasks.

    - Image-to-Text: example = {"conversation": [...]}, outputs input_ids + labels.
    - Text-to-Image/Editing: example = str (raw_text prompt), outputs input_ids + labels (with IGNORE_INDEX).
    """
    # assert image_token_idx is not None, "Must pass image_token_idx explicitly"
    # print(f"[DEBUG] image_token_idx = {image_token_idx}")
    if image_token_idx is None:
        tokenizer.add_tokens([image_token_str], special_tokens=True)
        image_token_idx = tokenizer.convert_tokens_to_ids(image_token_str)

    if isinstance(example, str):
        assert prompt_template is not None, \
            "prompt_template 不能为空（text2image/image-editing）"

        # 1) 构造 prompt
        #    直接在最前面加一个 <image> token，
        #    然后空一行，再拼原始文本
        prompt = f"{example.strip()}"
        # 用模板包装
        prompt = prompt_template["INSTRUCTION"].format(input=prompt)

        # 2) 用 tokenizer 编码（不要让 tokenizer 把 <image> 当成普通字符切分）
        #    一种简单做法：先去掉 tokenizer 里的特殊 token，再手动拼接
        text_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=(max_length - image_length) if max_length else None
        )
        # 把 <image> token id 插到最前面（或者你想要的位置）
        input_ids = [image_token_idx] * image_length + text_ids

        # 3) 如果超长，直接截断
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        # 4) attention_mask
        attention_mask = [1] * len(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # --- Image-to-text task: multi-turn conversation structure ---
    assert isinstance(example, dict) and "conversation" in example
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)
    is_multi_turn = len(example["conversation"]) > 1
    if is_multi_turn:
        assert input_ids_with_output

    input_ids, labels = [], []
    next_needs_bos_token = True

    for single_turn in example["conversation"]:
        input_text = single_turn["input"]

        # ==== Encode input ====
        if with_image_token and image_token_str in input_text:
            chunks = input_text.split(image_token_str)
            chunk_encoded = [tokenizer.encode(c, add_special_tokens=False) for c in chunks]
            assert len(chunk_encoded) >= 2
            input_encode = []
            for i, chunk in enumerate(chunk_encoded):
                input_encode.extend(chunk)
                if i < len(chunk_encoded) - 1:
                    input_encode.extend([image_token_idx] * image_length)
        else:
            input_encode = tokenizer.encode(input_text, add_special_tokens=False)

        if next_needs_bos_token:
            input_ids.extend(bos_token_id)
            labels.extend([IGNORE_INDEX] * len(bos_token_id))

        input_ids.extend(input_encode)
        labels.extend([IGNORE_INDEX] * len(input_encode))

        # ==== Encode output ====
        if input_ids_with_output and "output" in single_turn:
            output = single_turn["output"]
            output_with_loss = single_turn.get("output_with_loss", True)

            if with_image_token and image_token_str in output:
                chunks = output.split(image_token_str)
                chunk_encoded = [tokenizer.encode(c, add_special_tokens=False) for c in chunks]
                assert len(chunk_encoded) >= 2
                output_encode = []
                for i, chunk in enumerate(chunk_encoded):
                    output_encode.extend(chunk)
                    if i < len(chunk_encoded) - 1:
                        output_encode.extend([image_token_idx] * image_length)
            else:
                output_encode = tokenizer.encode(output, add_special_tokens=False)

            input_ids.extend(output_encode)
            if output_with_loss:
                labels.extend(output_encode.copy())
            else:
                labels.extend([IGNORE_INDEX] * len(output_encode))

            # ==== Append EOS ====
            if single_turn.get("need_eos_token", True):
                next_needs_bos_token = True
                input_ids.extend(eos_token_id)
                if output_with_loss:
                    labels.extend(eos_token_id.copy())
                else:
                    labels.extend([IGNORE_INDEX] * len(eos_token_id))
            else:
                next_needs_bos_token = False

            # ==== Append separator ====
            sep = single_turn.get("sep", "")
            if sep:
                sep_encoded = tokenizer.encode(sep, add_special_tokens=False)
                input_ids.extend(sep_encoded)
                labels.extend([IGNORE_INDEX] * len(sep_encoded))

    # ==== Truncation ====
    if max_length is not None and len(input_ids) > max_length:
        if truncation == "right":
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        elif truncation == "left":
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        else:
            raise ValueError("truncation must be 'left', 'right', or None")

    return {"input_ids": input_ids, "labels": labels}