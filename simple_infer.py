from PIL import Image
from pipeline import Pipeline

LORA_DIR = 'ckpt/normal-scene100-notext'
disable_prompts = LORA_DIR.endswith('-notext')
ppl = Pipeline(
    disable_prompts=disable_prompts,
    lora_ckpt=LORA_DIR,
    device='cuda',
    mixed_precision='fp16',
)
img = Image.open('test_Images/image_0027.png')
output_np_array = ppl(img, inference_step=5, target_mode='F')