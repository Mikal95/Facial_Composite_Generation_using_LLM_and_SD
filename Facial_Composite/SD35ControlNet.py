import random
import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models import SD3ControlNetModel
from diffusers.utils import load_image
from optimum.quanto import freeze, qfloat8, quantize, qint4

model_id = "stabilityai/stable-diffusion-3.5-medium"

# Load pipeline for either normal photo (tile), black-white drawing (canny), or inpainting.
controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Tile", torch_dtype=torch.float16)
#controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16)
#controlnet = SD3ControlNetModel.from_pretrained("alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1, torch_dtype=torch.float16)

pipe = StableDiffusion3ControlNetPipeline.from_pretrained( # For tile and canny
#pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained( # For inpainting
    model_id,
    torch_dtype=torch.float16,
    controlnet=controlnet
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

prompt3_1_corrector = ("Keep facial attributes. "
                       "more hooded eyelids. "
                       "goatee is split in half by patchy spot. "
                       "head gets narrower toward the top. "
                       "thicker lips. flatter ears. "
                       "wider jaw. fleshier nose. ")

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
                   "ears, "
                   "narrow face, "
                   "small lips, "
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

#control_image = load_image("reference_images/canny_new_my_face_3_18_768×1152_20_60_2510778634.png") # Use for canny reference -guided generation
#control_image = load_image("reference_images/my_face_3_18_768×1152_20_60_2510778634.png") # Use for tile or inpainting reference -guided generation
control_image = load_image("reference_images/crop_test_eye_my_face_3.png") # Eye image tile reference
#mask = load_image("reference_images/krita_mouth_my_face_3_18_768×1152_20_60_2510778634.png") # Mask image for inpainting

print("SD3.5 ControlNet running")
ccs_val = 0.0 # Reference image adherence strength, low strength recommended (<=0.5), is starting strength if auto_increase_ccs_val = True
auto_increase_ccs_val = True
guidance_scale = 20.0 # Prompt adherence strength
num_inference_steps = 60
#size = "768×1152" Portrait image resolution
#size = 1024 Square image resolution
batches = 5
img_per_batch = 5

eye_2_prompt_clip = "Macro close-up of a human eye looking at camera, male, pale skin, dark blue, almond, downturned, deeply set, average sized, very hooded upper eyelids that covers top of iris, natural reflections in the cornea, normal lighting, photorealistic"
eye_2_prompt_t5 = "Macro close-up of a human eye looking at camera, male, pale skin, dark blue, almond, downturned, deeply set, average sized, very hooded upper eyelids that covers top of iris, natural reflections in the cornea, normal lighting, photorealistic"
eye_2_neg_prompt_clip = "low quality, deformed iris, deformed pupil, bad eye, extra pupils"
eye_2_neg_prompt_t5 = "low quality, deformed iris, deformed pupil, bad eye, extra pupils"

for i in range(batches): # Number of batches
    print("iter_" + str(i) + "_ccs_" + str(ccs_val))
    for _ in range(img_per_batch): # Number of images per batch
        seed = random.randint(0, 4294967295)  # Manually creating random seed for record keeping
        image = pipe(
            # Face image prompts
            prompt=clip_prompt,
            prompt_3=prompt3_1,
            negative_prompt=negative_prompt,

            # Eye image prompts
            #prompt=eye_2_prompt_clip,
            #prompt_3=eye_2_prompt_t5,
            #negative_prompt=eye_2_neg_prompt_clip,
            #negative_prompt_3=eye_2_neg_prompt_t5,

            control_image=control_image,
            #control_mask=mask, # Mask image for inpainting
            controlnet_conditioning_scale=ccs_val, # For Tile and Canny -reference, sets reference image adherence strength
            # controlnet_conditioning_scale=0.95, # Inpainting strength, defaults to 1
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,

            # Square image resolution
            #height=size,
            #width=size,

            # Portrait image resolution
            height=1152,
            width=768,

            generator=torch.manual_seed(_) # Manually set random seed
        ).images[0]
        #image.save("controlnet_images/controlNetImage_" + str(i) + "_" + str(_) + ".png") # Tile
        #image.save("controlnet_images/canny_ControlNetImage_" + str(i) + "_" + str(_) + ".png") # Canny
        #image.save("controlnet_images/inpainting_ControlNetImage_" + str(_) + "_" + str(seed) + ".png") # Inpainting
        image.save("controlnet_images/controlNetImage_eye" + str(i) + "_" + str(_) + ".png") # Eye tile

    # Increase reference image adherence strength after every batch
    if auto_increase_ccs_val:
        ccs_val += 0.1
        ccs_val = round(ccs_val, 1)
