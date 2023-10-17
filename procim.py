from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from imutils.object_detection import non_max_suppression
import requests
from PIL import Image
import cv2
import numpy as np
import easyocr

def recognize( imager):    
    r_easy_ocr=reader.readtext(imager)
    imagev = cv2.cvtColor(imager, cv2.COLOR_GRAY2RGB)
    
    processor = TrOCRProcessor.from_pretrained("../trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("../trocr-large-handwritten")
    pixel_values = processor(imagev, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    processor2 = TrOCRProcessor.from_pretrained("../trocr-base-printed")
    model2 = VisionEncoderDecoderModel.from_pretrained("../trocr-base-printed")
    pixel_values2 = processor(imagev, return_tensors="pt").pixel_values
    generated_ids2 = model2.generate(pixel_values)
    generated_text2 = processor2.batch_decode(generated_ids2, skip_special_tokens=True)[0]
    return [('easyocr', r_easy_ocr), ('trocr-printed', generated_text2), ('trocr-handwritten', generated_text)]

def getrects( imager, rectangles):
    imgs=[]
    for r in rectangles:
        width=r[2]-r[0]
        height=r[3]-r[1]
        x=r[0]
        y=r[1]
        rectX = int(x)
        rectY = int(y)
        rectWidth = int(width)
        rectHeight = int(height)
        croppedImage = imager[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
        stri="tempImage" + str(x)  + str(y) + str(width) + str(height) + ".png"
        print (stri)
        cv2.imwrite(stri, croppedImage)
        imgs.append(croppedImage)
    return imgs
    
def process( image):
    gray = cv2.cvtColor(imageo, cv2.COLOR_RGB2GRAY)
    sharpen_kernel = np.array([[1,1,1], [7,-7,-5], [-1,-1,-1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    filename="thresh.png"
    cv2.imwrite(filename, thresh)
    return [('gray', gray), ('thresh', thresh), ('image', image)]
    
reader = easyocr.Reader(['es', 'en'],gpu = False) # load once only in memory.
imagename="carta.png"
image = Image.open(imagename).convert("RGB") #requests.get(url, stream=True).raw).convert("RGB")
imageo = cv2.imread(imagename)

model = cv2.dnn.readNet('../east//frozen_east_text_detection.pb')
height, width, _ = imageo.shape
new_height = (height//32)*32
new_width = (width//32)*32
h_ratio = height/new_height
w_ratio = width/new_width
print(h_ratio, w_ratio)
print(new_height, new_width)
blob = cv2.dnn.blobFromImage(imageo, 1, (new_width, new_height),(123.68, 116.78, 103.94), True, False)
model.setInput(blob)
print(model.getUnconnectedOutLayersNames())
(geometry, scores) = model.forward(model.getUnconnectedOutLayersNames())
print((geometry).shape, (scores).shape)
rectangles = [] #list 1
confidence_score = [] #list 2
for i in range(geometry.shape[2]): #iterate each pixel row by row to construct counding boxes. shape[2] pixel value = 96
    for j in range(0, geometry.shape[3]):  #shape[3] pixel value =160
        
        if scores[0][0][i][j] < 0.07:  #if score of pixel is less than threshold then we don't consider the pixel and we continue
            continue
            #otherwise obtain bounding box coordinates as follows
        bottom_x = int(j*4 + geometry[0][1][i][j])
        bottom_y = int(i*4 + geometry[0][2][i][j])
        top_x = int(j*4 - geometry[0][3][i][j])
        top_y = int(i*4 - geometry[0][0][i][j])        
        rectangles.append((top_x, top_y, bottom_x, bottom_y))
        confidence_score.append(float(scores[0][0][i][j]))
fin_boxes = non_max_suppression(np.array(rectangles), probs=confidence_score, overlapThresh=0.5)
img_copy = imageo.copy()
rectangles=[]
for (x1, y1, x2, y2) in fin_boxes:
    x1 = int(x1 * w_ratio)
    y1 = int(y1 * h_ratio)
    x2 = int(x2 * w_ratio)
    y2 = int(y2 * h_ratio)
    rectangles.append([x1, y1, x2, y2])
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0,255,0) , 2) #to draw rectangles on the image
cv2.imwrite("eastdetected.png", img_copy)
# sharp the edges or image.
for b in process(imageo):
    print (b[0])
    for c in getrects(b[1], rectangles):
        print (recognize(c))
    

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
        imageo,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
) 

print("Found {0} Faces!".format(len(faces)))
print(faces[0])
