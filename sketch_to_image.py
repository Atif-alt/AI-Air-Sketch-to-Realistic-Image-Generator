from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch
import os

print("Loading ControlNet Model...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float32
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

image = Image.open("sketches/sketch.png").convert("RGB")

prompt = input("Describe your sketch: ")

print("Generating image...")

result = pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=20
).images[0]

os.makedirs("outputs", exist_ok=True)

result.save("outputs/result.png")

print("Saved: outputs/result.png")