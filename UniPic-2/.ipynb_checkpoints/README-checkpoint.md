<div align="center">
  <h1><strong>Skywork-UniPic2</strong></h1>
</div>

<font size=7><div align='center' >  [[🤗 UniPic2 checkpoint](https://huggingface.co/collections/Skywork/skywork-unipic2-6899b9e1b038b24674d996fd)] [[📖 Tech Report](https://arxiv.org/abs/2508.xxxx)] </font> </div>

Welcome to the Skywork-UniPic2 repository! This repository contains the model weights and implementation of our enhanced unified model that integrates image understanding, text-to-image generation, and image editing capabilities with improved performance and efficiency.

<div align="center">
  <img src="assets/teaser.png" alt="Skywork UniPic2 Teaser" width="90%">
</div>

## What's New in UniPic2

- **Enhanced Architecture**: Improved model architecture based on Stable Diffusion 3.5 and Flux
- **Better Image Quality**: Advanced preprocessing and generation techniques
- **Unified Framework**: Seamless integration of text-to-image and image editing tasks
- **Optimized Inference**: More efficient inference pipeline with better memory management
- **Advanced Parameters**: Fine-grained control over generation quality and style

## Evaluation

<p align="center"><strong>Performance Overview</strong></p>

<div align="center">
  <img src="assets/main_comparison_v2.png" alt="Main comparison" width="800"/>
</div>

<p align="center"><strong>GenEval</strong></p>
<div align="center">



| Model | Single | Two | Count | Color | Position | Attr | **Overall** |
|:------|:------:|:---:|:-----:|:-----:|:--------:|:----:|:-----------:|
| **Previous Models** |||||||||
| SD3‑medium      | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |
| FLUX.1‑dev      | 0.99 | 0.81 | 0.79 | 0.74 | 0.20 | 0.47 | 0.67 |
| OmniGen2        | 1.00 | 0.95 | 0.64 | 0.88 | 0.55 | 0.76 | 0.80 |
| **Skywork UniPic** | 0.98 | 0.92 | 0.74 | 0.91 | 0.89 | 0.72 | 0.86 |
| **Skywork UniPic2** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

</div>

<p align="center"><strong>DPG‑Bench</strong></p>
<div align="center">

| Model | Global | Entity | Attribute | Relation | Other | **Overall** |
|:------|:------:|:------:|:---------:|:--------:|:-----:|:-----------:|
| **Previous Models** |||||||
| SD3‑medium       | 87.90 | 91.01 | 88.83 | 80.70 | 88.68 | 84.08 |
| FLUX.1‑dev       | 82.10 | 89.50 | 88.70 | 91.10 | 89.40 | 84.00 |
| OmniGen2         | 88.81 | 88.83 | 90.18 | 89.37 | 90.27 | 83.57 |
| **Skywork UniPic** | 89.65 | 87.78 | 90.84 | 91.89 | 91.95 | 85.50 |
| **Skywork UniPic2** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

</div>

<p align="center"><strong>GEdit‑Bench‑EN</strong></p>
<div align="center">

| Model | SC ↑ | PQ ↑ | **Overall ↑** |
|:------|:----:|:----:|:--------------:|
| **Specialized Editing Models** ||||
| Step1X‑Edit      | 7.09 | 6.76 | 6.70 |
| **Unified Models** ||||
| OmniGen2         | 7.16 | 6.77 | 6.41 |
| BAGEL            | 7.36 | 6.83 | 6.52 |
| **Skywork UniPic** | 6.72 | 6.18 | 5.83 |
| **Skywork UniPic2** | **TBD** | **TBD** | **TBD** |

</div>
</div>

## Usage

### 📦 Required Packages
Create virtual environment and install dependencies with pip:
```shell
conda create -n unipic_v2 python==3.10.14
conda activate unipic_v2
pip install -r requirements.txt
```

### 📥 Checkpoints

Download the model checkpoints from [🤗 Skywork UniPic2](https://huggingface.co/collections/Skywork/skywork-unipic2-6899b9e1b038b24674d996fd),
It is recommended to use the following command to download the checkpoints
```bash
# pip install -U "huggingface_hub[cli]"
huggingface-cli download Skywork/Skywork-UniPic2 --local-dir checkpoint --repo-type model
```

### 🚀 Quick Start with Scripts

We provide four standalone scripts for different inference modes:

#### Method 1: SD3.5M Kontext

**Text-to-Image Generation:**
```bash
python scripts/sd35m_text2image.py \
    --checkpoint_path /path/to/unipicv2_sd_3_5m_kontext \
    --prompt "a pig with wings and a top hat flying over a happy futuristic scifi city" \
    --output text2image.png \
    --height 512 --width 384 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --seed 42
```

**Image Editing:**
```bash
python scripts/sd35m_image_editing.py \
    --checkpoint_path /path/to/unipicv2_sd_3_5m_kontext \
    --input_image text2image.png \
    --prompt "remove the pig's hat" \
    --output image_editing.png \
    --image_size 512 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --seed 42
```

#### Method 2: Qwen2.5-VL + SD3.5M Kontext

**Text-to-Image Generation:**
```bash
python scripts/qwen_text2image.py \
    --checkpoint_path /path/to/unipicv2_qwen2_5_vl_7b_sd_3_5m_kontext \
    --prompt "a pig with wings and a top hat flying over a happy futuristic scifi city" \
    --output qwen_text2image.png \
    --height 512 --width 384 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --seed 42
```

**Image Editing with Vision Input:**
```bash
python scripts/qwen_image_editing.py \
    --checkpoint_path /path/to/unipicv2_qwen2_5_vl_7b_sd_3_5m_kontext \
    --input_image input_image.png \
    --prompt "remove the pig's hat" \
    --output qwen_image_editing.png \
    --image_size 512 \
    --num_inference_steps 50 \
    --guidance_scale 3.5 \
    --seed 42
```


## Citation

If you use Skywork-UniPic2 in your research, please cite:
```

```

## 📜 License
This project is licensed under [MIT License](LICENSE).