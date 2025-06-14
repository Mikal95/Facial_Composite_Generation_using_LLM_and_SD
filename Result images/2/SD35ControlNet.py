import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from transformers import T5EncoderModel, BitsAndBytesConfig
from optimum.quanto import freeze, qfloat8, quantize, qint4
import numpy as np



model_id = "stabilityai/stable-diffusion-3.5-medium"

# load pipeline
controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Tile",
                                                torch_dtype=torch.float16
                                                )
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    controlnet=controlnet,
)
pipe.to("cuda")

quantize(pipe.transformer, weights=qfloat8)
freeze(pipe.transformer)

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

control_image = load_image("my_face_13_19_768×1152_13_28_3236556164.png")

print("SD3.5 ControlNet running")
ccs_val = 0.0
guidance_scale = 15.0
num_inference_steps = 28
size = "768×1152"

for i in range(5):
    print("iter_" + str(i) + "_ccs_" + str(ccs_val))
    for _ in range(10):
        image = pipe(
            prompt=clip_prompt,
            prompt_3=prompt3_1,
            negative_prompt=negative_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=ccs_val,
            guidance_scale=guidance_scale,
            height=1152,
            width=768,
            generator=torch.manual_seed(_)
        ).images[0]
        image.save("controlNetImage" + str(i) + "_" + str(_) + ".png")

    ccs_val += 0.1
    ccs_val = round(ccs_val, 1)
        #  low control strength (<0.4)

# controlNetImage3_ uses quantized model
# controlNetImage4_ uses quantized text_encoder_3
# controlNetImage5_ uses quantized text_encoder and text_encoder_3
# controlNetImage6_ uses quantized model and text_encoder_3
"""
100%|██████████| 28/28 [02:31<00:00,  5.42s/it]
100%|██████████| 28/28 [02:31<00:00,  5.41s/it]
100%|██████████| 28/28 [02:28<00:00,  5.31s/it]
100%|██████████| 28/28 [02:29<00:00,  5.34s/it]
100%|██████████| 28/28 [02:30<00:00,  5.38s/it]
100%|██████████| 28/28 [02:26<00:00,  5.23s/it]
100%|██████████| 28/28 [02:28<00:00,  5.31s/it]
100%|██████████| 28/28 [02:28<00:00,  5.29s/it]
100%|██████████| 28/28 [02:36<00:00,  5.60s/it]
100%|██████████| 28/28 [02:27<00:00,  5.25s/it]

100%|██████████| 28/28 [01:58<00:00,  4.23s/it]
100%|██████████| 28/28 [01:59<00:00,  4.28s/it]
100%|██████████| 28/28 [02:03<00:00,  4.42s/it]
100%|██████████| 28/28 [02:01<00:00,  4.36s/it]
100%|██████████| 28/28 [02:08<00:00,  4.59s/it]
100%|██████████| 28/28 [01:59<00:00,  4.28s/it]
100%|██████████| 28/28 [02:02<00:00,  4.39s/it]
100%|██████████| 28/28 [02:00<00:00,  4.30s/it]
100%|██████████| 28/28 [02:06<00:00,  4.51s/it]
100%|██████████| 28/28 [02:01<00:00,  4.33s/it]
"""
# controlNetImage8_ uses controlnet_conditioning_scale=0.0 and cfg=15
# controlNetImage8_ uses controlnet_conditioning_scale=0.0 and cfg=7

# controlNetImage10_ uses controlnet_conditioning_scale=0.0 and cfg=10
"""
100%|██████████| 28/28 [01:45<00:00,  3.76s/it]
100%|██████████| 28/28 [01:50<00:00,  3.94s/it]
100%|██████████| 28/28 [01:45<00:00,  3.78s/it]
100%|██████████| 28/28 [01:45<00:00,  3.78s/it]
100%|██████████| 28/28 [01:46<00:00,  3.79s/it]
100%|██████████| 28/28 [01:50<00:00,  3.94s/it]
100%|██████████| 28/28 [01:49<00:00,  3.91s/it]
100%|██████████| 28/28 [01:54<00:00,  4.10s/it]
100%|██████████| 28/28 [01:46<00:00,  3.81s/it]
100%|██████████| 28/28 [01:50<00:00,  3.94s/it]

100%|██████████| 28/28 [01:14<00:00,  2.68s/it]
100%|██████████| 28/28 [01:10<00:00,  2.53s/it]
100%|██████████| 28/28 [01:09<00:00,  2.48s/it]
100%|██████████| 28/28 [01:08<00:00,  2.44s/it]
100%|██████████| 28/28 [01:12<00:00,  2.58s/it]
100%|██████████| 28/28 [01:07<00:00,  2.43s/it]
100%|██████████| 28/28 [01:08<00:00,  2.44s/it]
100%|██████████| 28/28 [01:10<00:00,  2.51s/it]
100%|██████████| 28/28 [01:13<00:00,  2.63s/it]
100%|██████████| 28/28 [01:09<00:00,  2.48s/it]
"""

# controlNetImage11_ uses controlnet_conditioning_scale=0.0, quantization qint4 and cfg=10