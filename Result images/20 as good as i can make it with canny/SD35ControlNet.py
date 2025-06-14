import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image
from transformers import T5EncoderModel, BitsAndBytesConfig
from optimum.quanto import freeze, qfloat8, quantize, qint4
import numpy as np

model_id = "stabilityai/stable-diffusion-3.5-medium"

controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)

pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    controlnet=controlnet,
)
pipe.to("cuda")

quantize(pipe.transformer, weights=qfloat8)
freeze(pipe.transformer)

prompt3_1 = ("mugshot of a norwegian man. "
          "facing viewer. "
             "looking at viewer. "
           "blank, white background. "
           "neutral expression. "
           "normal lighting. "
            "age late 20s. "
           "pale skin. "
           "wide oval shaped face. "
             "wide square jaw. "
             "square chin. "
             "round narrow forehead. "
           "very short side swept quiff. "
           "dark blonde hair. "
           "straight hairline. "
           "big smooth forehead. "
           "no ears. "
           "dark brown eyebrows with soft angle. "
           "eyes are dark blue, almond, downturned, deepset, average sized with very hooded eyelids. "
           "normal eye gap. "
           "prominent cheekbones. "
             "full, chubby face cheeks, "
           "nose is straight and slightly raised. "
           "very big heart shaped lips with downturned corners. "
           "patchy, black, thin chin strap beard. "
          "slightly overweight. "
             "slightly overweight. "
             "photorealistic. "
             )

clip_prompt = ("mugshot of a norwegian man. "
           "wide oval shaped face. "
            "square jaw. "
           "dark blonde, very short side swept quiff. "
            "round narrow forehead. "
            "prominent cheekbones. "
            "full, chubby face cheeks, "
            "square chin. "
            "average sized dark blue eyes with very hooded eyelids. "
           "big heart shaped lips with downturned corners. "
           "patchy, black, thin chin strap beard. "
               )

negative_prompt = ("wide forehead, "
                   "narrow face, "
                   "mustache, "
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
                   "blurry, "
                   "monochrome, "
                   "saturated, "
                   "bad image quality, "
                   "cartoonish, "
                   "deformed, "
                   "disfigured, "
                   "bad, "
                   "deformed iris, "
                   "deformed pupils")

control_image = load_image("reference_images/canny_new_my_face_3_18_768×1152_20_60_2510778634.png")

print("SD3.5 ControlNet running")
ccs_val = 0.0
guidance_scale = 20.0
num_inference_steps = 60
size = "768×1152"

for i in range(6):
    print("iter_" + str(i) + "_ccs_" + str(ccs_val))
    for _ in range(20):
        image = pipe(
            prompt=clip_prompt,
            prompt_3=prompt3_1,
            negative_prompt=negative_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=ccs_val,
            num_inference_steps=num_inference_steps, # was 28
            guidance_scale=guidance_scale,
            height=1152,
            width=768,
            generator=torch.manual_seed(_)
        ).images[0]
        image.save("controlnet_images/cannyControlNetImage_" + str(i) + "_" + str(_) + ".png")

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