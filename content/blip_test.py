import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

def generate_caption(image_path, prompt=None):
    # Load model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Open image
    image = Image.open(image_path).convert('RGB')

    # Preprocess input
    inputs = processor(image, text=prompt, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs, max_length=64, num_beams=5)
    
    # Decode output
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP Image Captioning")
    parser.add_argument("image_path", type=str, help="Path to the local image file")
    parser.add_argument("--prompt", type=str, default="a photo of", help="Optional prompt to guide caption generation")

    args = parser.parse_args()
    caption = generate_caption(args.image_path, args.prompt)
    print(f"Caption:\n{caption}")
