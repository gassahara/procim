from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import cv2
import easyocr

processor = TrOCRProcessor.from_pretrained("../trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("../trocr-base-printed")

# load image from the IAM dataset

url = "carta.png"
imagename="carta.png"
image = Image.open(imagename).convert("RGB")
imageo = cv2.imread(imagename)
#imagev = cv2.cvtColor(imageo, cv2.COLOR_RGB2GRAY)
pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text);
reader = easyocr.Reader(['en'],gpu = False) # load once only in memory.
gray = cv2.cvtColor(imageo, cv2.COLOR_RGB2GRAY)
r_easy_ocr=reader.readtext(gray)
print(r_easy_ocr)
