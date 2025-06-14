import random

import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
from transformers.agents import prompts

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "stabilityai/stable-diffusion-3.5-medium"
text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=text_encoder,
    device_map="balanced",
    torch_dtype=torch.float16,
)

prompt3_1 = ("a mugshot of a norwegian man. "
          "facing viewer. "
           "blank, white background. "
           "neutral expression. "
           "normal lighting. "
            "age late 20s. "
           "pale skin. "
           "rectangle face shape. "
           "brushed up crew cut hairstyle. "
           "dark blonde hair. "
           "straight hairline, big gap between the forehead and the hair. "
           "smooth forehead. "
           "flat ears. "
           "eyebrows are light brown, with soft angle. "
           "eyes are dark blue, almond, downturned, deepset, average sized with very hooded eyelids. "
           "normal gap between the eyes. "
           "cheekbones are prominent, giving the face a slightly angular appearance. "
             "full, chubby cheeks, "
           "nose is straight and slightly raised. "
           "big, heart shaped lips with downturned corners. "
           "jawline is defined, with a slight square shape. "
           "pointy chin. "
           "patchy, very thin, black chin strap beard. "
          "slightly overweight. ")

clip_prompt = ("mugshot of a norwegian man. "
          "facing viewer. "
           "white background. "
           "neutral expression. "
           "normal lighting. "
           "late 20s. "
           "pale skin. "
           "rectangle face shape. "
           "dark blonde, brushed up crew cut hairstyle. "
           "straight hairline. "
           "smooth forehead. "
           "flat ears. "
           "dark brown eyebrows, with soft angle. "
           "eyes are dark blue, almond, downturned, deepset, and average sized with very hooded eyelids. "
           "normal gap between the eyes. "
           "prominent cheekbones. "
           "full, chubby cheeks, "
           "slightly raised and straight nose. "
           "big heart shaped lips with downturned corners. "
           "defined jawline, slight square shape. "
           "pointy chin. "
           "patchy, very thin, black chin strap beard. "
           "slightly overweight. ")

negative_prompt = ("buzz cut hairstyle, "
                   "mustache, "
                   "receding hairline, "
                   "skinny, "
                   "colored light, "
                   "colored background, "
                   "pixel art, "
                   "overexposed, "
                   "underexposed, "
                   "low quality, "
                   "grainy, "
                   "blurry, "
                   "monochrome, "
                   "saturated, "
                   "bad image quality, cartoonish, 3D render, logo")
print("Stable Diffusion 3.5 running")
guidance_scale = 13.0
num_inference_steps = 28
size = "768×1152"
#size = 1024
for _ in range(20):
    seed = random.randint(0, 4294967295)  # 4294967296
    image = pipe(prompt=clip_prompt,
                 prompt_3=prompt3_1,
                 negative_prompt=negative_prompt,
                 generator=torch.manual_seed(_),
                 guidance_scale=guidance_scale,
                 num_inference_steps=num_inference_steps,
                 max_sequence_length=256 * 2,
                 height=1152,
                 width=768,
                 #skip_guidance_layers=[7, 8, 9], # recommended by StabilityAI for Stable Diffusion 3.5 Medium
                 #skip_layer_guidance_scale=7.0
                 ).images[0]
    file_name = "my_face_13_" + str(_) + "_" + str(size) + "_" + str(int(guidance_scale)) + "_" + str(num_inference_steps) + "_" + str(seed) + ".png"
    image.save(file_name)

# THIS WORKS!!, TEST USING THIS AND CHANGE BACK TO ORIGINAL WHEN EXPECTING BEST RESULT
# my_face_7_ uses portrait aspect ratio 768×1152, generally better face quality that square 1024x1024, similar to findings in:
# https://civitai.com/articles/8460/stable-diffusion-35-large-a-short-experiment-on-quality-resolution-and-aspect-ratio

# my_face_9_ uses portrait and SGL