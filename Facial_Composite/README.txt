Credits:
Built with Llama
Powered by Stability AI


Install instructions:

1. Use python version 3.12

2. If you are using a Nvidia GPU, install CUDA toolkit version 12.6 from https://developer.nvidia.com/cuda-12-6-0-download-archive

3. Find correct PyTorch version using install instructions from https://pytorch.org/get-started/locally/ then run the pip command:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126



Usage instructions:
(This project is two separate (LLM and SD) projects combined into one,
checks were made to see models were running,
but technical problems could arise)

1. Image generation:
Stable Diffusion 3.5 Medium was used to generate mugshot images from text prompts,
but requires permission and signed license agreement to download from HuggingFace using the HuggingFace CLI library.
Link: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

Your original long prompt goes into text encoder 3 (T5-xxl, max length 256 tokens),
shorter version of mugshot prompt (words that you think are most important) goes into text encoder 1 (OpenCLIP-ViT/G, max length 77 tokens).

1.1. For face image generation, SD35.py already has
face description prompts,
guidance scale (20.0),
number of inference steps (60),
image size (1024 x 1024 pixels),
and number of images to generate (20),
change to what you want and run SD35.py.
Images generated using SD35.py will be saved in the Generated_images -folder.

1.2. For face image generation using ControlNet reference,
comment/decomment highlighted code lines in SD35ControlNet.py belonging either to Tile,
Canny or Inpainting depending on what you want to use,
put reference images you want to use into 'reference_images' -folder,
then run SD35ControlNet.py.
Images generated using SD35ControlNet.py will be saved in the controlnet_images -folder.


2. Text generation:
Face description interpretation and merging was done using Llama 3.1 8B Instruct with
4bit quantization_config using BitsAndBytes library,
torch.bfloat16 precision,
the Accelerate library with device_map=“auto”,
torch.compile with "reduce-overhead" mode and 'fullgraph',
and GUI was made using Gradio.
Llama 3.1 8B Instruct requires permission and signed license agreement to download from HuggingFace using the HuggingFace CLI library.
Link: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

The LLM is meant to merge facial characteristic descriptions from the document 'Facial Image Comparison Feature List
for Morphological Analysis' (V2.0, not in draft status) from the Facial Identification Scientific Working Group (FISWG),
all the LLM's instructions are in the system prompt in LLM_Gradio.py,
when running the program write your facial characteristic descriptions that are to be merged into user query.

To use the LLM run LLM_Gradio.py and click the URL appearing in terminal (http://127.0.0.1:7860),
a web page with a chat interface should appear, and you can now write to the LLM.

Alternatively you can run it by entering "gradio LLM_Gradio.py" into terminal,
now you can edit LLM_Gradio.py and changes will take effect live in the Gradio GUI,
without having to close the program and reload LLM.
Exit the program by pressing CTRL-C in terminal.



