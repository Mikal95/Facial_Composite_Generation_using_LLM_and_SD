import random
import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.utils import load_image
from transformers import T5EncoderModel, BitsAndBytesConfig
from optimum.quanto import freeze, qfloat8, quantize, qint4
import numpy as np

model_id = "stabilityai/stable-diffusion-3.5-medium"

# load pipeline
controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Tile", torch_dtype=torch.float16)

pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    controlnet=controlnet,
)
pipe.to("cuda")

quantize(pipe.transformer, weights=qfloat8)
freeze(pipe.transformer)

control_image = load_image("crop_test_eye_my_face_3.png")

print("SD3.5 ControlNet running")
ccs_val = 0.0
guidance_scale = 20.0
num_inference_steps = 60
#size = "768×1152"
size = 1024

eye_prompt_clip = "Macro close-up of a human eye looking at camera, dark blue, almond, downturned, deeply set, average sized, very hooded upper eyelids, natural reflections in the cornea, pale skin, photorealistic"
eye_prompt_t5 = "Macro close-up of a human eye looking at camera, dark blue, almond, downturned, deeply set, average sized, very hooded upper eyelids, natural reflections in the cornea, pale skin, photorealistic"
eye_neg_prompt_clip = "low quality, deformed iris, deformed pupil, bad eye"
eye_neg_prompt_t5 = "low quality, deformed iris, deformed pupil, bad eye"

for i in range(5):
    # print("iter_" + str(i) + "_ccs_" + str(ccs_val))
    for _ in range(5):
        seed = random.randint(0, 4294967295)  # 4294967296
        image = pipe(
            control_image=control_image,
            prompt=eye_prompt_clip,
            prompt_3=eye_prompt_t5,
            negative_prompt=eye_neg_prompt_clip,
            negative_prompt_3=eye_neg_prompt_t5,
            controlnet_conditioning_scale=ccs_val,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=size,
            width=size,
            generator=torch.manual_seed(_)
        ).images[0]
        #image.save("controlnet_images/controlNetImage_" + str(i) + "_" + str(_) + ".png")
        #image.save("controlnet_images/cannyControlNetImage_" + str(i) + "_" + str(_) + ".png")
        #image.save("controlnet_images/inpainting_ControlNetImage_" + str(_) + "_" + str(seed) + ".png")
        image.save("controlnet_images/controlNetImage_eye" + str(i) + "_" + str(_) + ".png")

    ccs_val += 0.1
    ccs_val = round(ccs_val, 1)
        #  low control strength (<0.4)