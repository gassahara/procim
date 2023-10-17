from transformers import AutoImageProcessor, BeitForMaskedImageModeling, AutoModelForImageClassification
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open('./f.png').convert('RGB')

processor = AutoImageProcessor.from_pretrained("/media/user/TOSHIBA EXT/comp/dit-base")
model = BeitForMaskedImageModeling.from_pretrained("/media/user/TOSHIBA EXT/comp/dit-base")
#model = AutoModelForImageClassification.from_pretrained("/media/user/TOSHIBA EXT/comp/dit-base")

num_patches = (model.config.image_size // model.config.patch_size) ** 2
pixel_values = processor(images=image, return_tensors="pt").pixel_values
# create random boolean mask of shape (batch_size, num_patches)
bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
print(outputs)
#loss, logits = outputs.loss, outputs.logits

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
print(":", logits)
print("|", logits.argmax(-1))
predicted_label = logits.argmax(-1)[0][0].item()
print(predicted_label)
print(model.config.id2label[predicted_label])
