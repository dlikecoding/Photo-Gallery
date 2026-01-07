import sys, json, torch, os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

model_path = os.environ.get("BLIP_MODEL_PATH")
# processor_path = os.environ.get("CONFIG_PATH")

# model_path = "./blip_image_captioning"

processor = BlipProcessor.from_pretrained(model_path, use_fast=True)
model = BlipForConditionalGeneration.from_pretrained(model_path)

# model = torch.load(model_path, map_location='cpu')
# model.eval()

try:
    for line in sys.stdin:
        task = json.loads(line)
        id = task.get("id")

        img_url = task.get("path")
        raw_image = Image.open(img_url).convert('RGB')
        
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=120)
        result = processor.decode(out[0], skip_special_tokens=True)

        # Response based on task
        result = { "media_id": id, "caption": result }
        
        print(json.dumps(result))
        sys.stdout.flush()

except json.JSONDecodeError:
    print("Invalid JSON received")
    sys.stdout.flush()
