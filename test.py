import vllm
from vllm import LLM
import PIL.Image

llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Refer to the HuggingFace repo for the correct format to use
prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:"

# Load the image using PIL.Image
image = PIL.Image.open("252219.png")

# Single prompt inference
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
})

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)


# Batch inference
image_1 = PIL.Image.open("252219.png")
image_2 = PIL.Image.open("252219.png")

outputs = llm.generate(
    [
        {
            "prompt": "USER: <image>\nWhat is the content of this image?\nASSISTANT:",
            "multi_modal_data": {"image": image_1},
        },
        {
            "prompt": "USER: <image>\nWhat's the color of this plate?\nASSISTANT:",
            "multi_modal_data": {"image": image_2},
        }
    ]
)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)