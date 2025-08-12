#!/usr/bin/env python3
"""
Skywork-UniPic2 mq Image-to-Text Generation
Usage: python unipic2_mq_i2t.py --model_path /path/to/Qwen2.5-VL --image_path input.jpg --prompt "Describe the image" --output output.txt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

def main():
    parser = argparse.ArgumentParser(description='SD3.5M Kontext Image-to-Text Generation')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the Qwen2.5-VL model')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--prompt', type=str, default='Describe the image in detail.',
                      help='Prompt for image description')
    parser.add_argument('--output', type=str, default='image_description.txt',
                      help='Output text file path')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                      help='Maximum number of new tokens to generate')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    # Load model components - using Qwen2.5-VL specific classes
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    # Modify chat template (same as in your t2i implementation)
    processor.chat_template = processor.chat_template.replace(
        "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
        "")
    
    print(f"Processing image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    
    # Process chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Prepare inputs - using Qwen2.5-VL processor
    inputs = processor(
        text=[text],
        images=[image],  # Pass the image directly
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    # Generate description
    print("Generating image description...")
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=args.max_new_tokens
    )
    
    # Process output
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Remove the input prompt from output if present
    if text in generated_text:
        generated_text = generated_text.replace(text, "").strip()
    
    # Save output
    with open(args.output, 'w') as f:
        f.write(generated_text)
    
    print(f"Image description saved to: {args.output}")
    print("Generated description:")
    print(generated_text)

if __name__ == "__main__":
    main()
