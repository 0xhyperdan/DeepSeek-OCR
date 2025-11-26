import os

import torch
from transformers import AutoModel, AutoTokenizer

# 固定用第 0 张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 减少显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
# 适配 Tesla P4 8GB 显卡
model = (
    AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float16,  # ★ 用 half，P4 支持
        _attn_implementation="eager",  # 或 "sdpa"，P4 用不了 FA2 也无所谓
    )
    .to("cuda")
    .eval()
)
model = model.eval().cuda().to(torch.bfloat16)


# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......
prompt = "<image>\nFree OCR."
image_file = "./images/img1.jpg"
output_path = "./images/output/"


# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode=False, save_results = True, test_compress = True)

with torch.inference_mode():  # 关闭梯度，省显存
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=640,
        image_size=640,
        crop_mode=False,
        save_results=True,
        test_compress=True,
    )

# print(res)
