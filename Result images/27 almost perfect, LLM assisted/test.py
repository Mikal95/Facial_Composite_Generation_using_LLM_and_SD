import random

import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig
from transformers.agents import prompts
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler

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

t5_prompt = ("Mugshot of a norwegian man. "
          "Facing viewer. "
             "Looking at viewer. "
           "Neutral expression. "
            "Age late 20s. "
           "Pale skin. "
           "Triangle face shape. "
             "Wide and square jaw with vertical mandibular angle, slightly receding chin. "
             "Pointy double chin. "
             "Very short and dark blonde side swept quiff. "
              "Gradually tapering forehead, narrowing towards the crown. "
              "Rounded hairline. "
           "Dark brown eyebrows with soft angle. "
           "Eyes are dark blue, almond, downturned, deeply set, average sized with very hooded eyelids. "
           "Normal eye gap. "
            "Slightly dark circles under eyes. "
           "High and prominent cheekbones. "
              "Hollow cheeks. "
           "Fleshy, straight and slightly raised nose with bulbous tip and deeply grooved philtrum. "
           "Full, wide and pink M-shaped lips. "
           "Patchy, black, thin chin strap beard. "
          "Slightly overweight. "
             "Photorealistic. "
             "White background. "
             "Normal lighting. "
)

clip_prompt = ("Mugshot of a norwegian man. "
            "Age late 20s. "
           "Pale skin. "
           "Triangle face shape. "
             "Wide square jaw. "
             "Pointy chin. "
             "Dark blonde quiff. "
                "Rounded forehead. "
           "Dark blue eyes with hooded eyelids. "
           "Prominent cheekbones. "
              "Hollow cheeks. "
           "Full M-shaped lips. "
           "Patchy black beard. "
)

clip_negative_prompt = ("overexposed, "
                     "low quality, "
                     "monochrome, "
                     "cartoon, "
                     "receding hairline, "
                     "mustache, "
                     "soul patch, "
                     "very thick beard, "
                     "hollow temples, "
                        "thin lips, "
                     "protruding ears, "
                     "non-hooded eyelids, "
                     "forehead wrinkles, "
                       "deformed iris, "
                       "deformed pupils, "
                       "open mouth, "
                       "disfigured, "
                       "asymmetrical, "
                       "bad eyes, "
                       "asymmetric ears"
)

t5_negative_prompt = ("overexposed, "
                     "low quality, "
                     "monochrome, "
                     "colored background, "
                     "cartoon, "
                     "receding hairline, "
                     "mustache, "
                     "soul patch, "
                     "very thick beard, "
                     "hollow temples, "
                        "thin and narrow lips, "
                     "protruding ears, "
                     "non-hooded eyelids, "
                     "forehead wrinkles, "
                       "deformed iris, "
                       "deformed pupils, "
                       "open mouth, "
                       "disfigured, "
                       "asymmetrical, "
                       "bad eyes, "
                       "asymmetric ears, "
                       "high mandibular angle, "
                       "low cheekbones, "
                       "skinny nose, "
                       "long neck, "
                       "big nostrils, "
                       "smile, "
                       "acne, "
                       "heart shaped face, "
                       "oval shaped face, "
                       "uneven skin tone "
                       "shiny skin, "
                       "narrow jaw, "
                       "nsfw, "
                       "nude, "
                       "nudity, "
                       "uncensored, "
                       "explicit content"
)

print("Stable Diffusion 3.5 running")
guidance_scale = 20.0 # 15.0, 20.0
num_inference_steps = 60 # 28, 50, 60
#size = "768×1152"
size = 1024
for _ in range(20):
    #seed = random.randint(0, 4294967295)  # 4294967296
    seed = _
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt=clip_prompt,
                 prompt_3=t5_prompt,
                 negative_prompt=clip_negative_prompt,
                 negative_prompt_3=t5_negative_prompt,
                 generator=generator,
                 guidance_scale=guidance_scale,
                 num_inference_steps=num_inference_steps,
                 max_sequence_length=256 * 2,
                 height=1024,
                 width=1024,
                 ).images[0]
    file_name = "proper_llm_my_face_2_" + str(_) + "_" + str(size) + "_" + str(int(guidance_scale)) + "_" + str(num_inference_steps) + "_" + str(seed) + ".png"
    image.save(file_name)
