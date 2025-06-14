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
           "diamond face shape. "
           "very short side swept quiff. "
           "dark blonde hair. "
           "straight hairline, big gap between the forehead and the hair. "
           "smooth forehead. "
           "flat ears. "
           "dark brown eyebrows with soft angle. "
           "eyes are dark blue, almond, downturned, deepset, average sized with very hooded eyelids. "
           "normal eye gap. "
           "cheekbones are prominent, giving the face a slightly angular appearance. "
             "full, chubby face cheeks, "
           "nose is straight and slightly raised. "
           "very big, heart shaped lips with downturned corners. "
             "triangular jaw. "
           "patchy, black, thin chin strap beard. "
          "slightly overweight. "
             "sharp image quality. "
             )

clip_prompt = ("mugshot of a norwegian man. "
           "diamond face shape. "
           "dark blonde, very short side swept quiff. "
           "dark brown eyebrows with soft angle. "
           "eyes are dark blue, almond, downturned, deepset, average sized with very hooded eyelids. "
           "full, chubby face cheeks, "
           "very big heart shaped lips with downturned corners. "
           "patchy, black, thin chin strap beard. "
               )

# glossy
negative_prompt = ("mustache, "
                   "receding hairline, "
                   "skinny, "
                   "soul patch, "
                   "very thick beard, "
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
guidance_scale = 15.0
num_inference_steps = 50 # 28
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
                 ).images[0]
    file_name = "my_face_20_" + str(_) + "_" + str(size) + "_" + str(int(guidance_scale)) + "_" + str(num_inference_steps) + "_" + str(seed) + ".png"
    image.save(file_name)
