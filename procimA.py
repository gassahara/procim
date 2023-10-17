from imutils.object_detection import non_max_suppression
from PIL import Image
import cv2
import numpy as np
import easyocr
import math
import distinctipy

def process( imageA):
    hsv = cv2.cvtColor(imageA, cv2.COLOR_RGB2HSV)
    lower_green = np.array([10, 10, 0])
    upper_green = np.array([360, 360, 100])
    min_hue = 0
    max_hue = 30
    min_sat = 0
    max_sat = 30
#    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.inRange(hsv, (min_hue, min_sat, 0), (max_hue, max_sat, 255))
    mask = 255-mask
    filename="hsvmask.png"
    cv2.imwrite(filename, mask)
    result = cv2.bitwise_and(255-imageA, 255-imageA, mask=mask)
    result=255-result
    filename="hsvmasked.png"
    cv2.imwrite(filename, result)
    gray = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    sharpen_kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    edge_kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
    edges = cv2.filter2D(gray, -1, edge_kernel)
    thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    i=10
    contourB=thresh.copy()
    contr=[]
    colors = distinctipy.get_colors(200)
    contour=result.copy()
    while (i<200):
        _, binary = cv2.threshold(contour, i, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#        cv2.drawContours(contourB, contours, -1, (0, 0, 0), 2)
        cv2.drawContours(contourB, contours, -1, colors[i], 1)
        contr.append(contours)
        i=i+1
    thresh_edge = cv2.threshold(contourB, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    canny = cv2.Canny(thresh_edge,25,50)
    thresh_edge = cv2.cvtColor(thresh_edge, cv2.COLOR_GRAY2RGB)
    canny_color = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    filename="thresh.png"
    cv2.imwrite(filename, thresh)
    filename="canny.png"
    cv2.imwrite(filename, canny_color)
    filename="edges.png"
    cv2.imwrite(filename, thresh_edge)
    filename="contours.png"
    cv2.imwrite(filename, contour)
    return {'gray': gray, 'thresh': thresh, 'thresh_edge': thresh_edge, 'canny': canny, 'edges': edges, 'contour': contour, 'mask': mask_color}
    
imagename="carta2.png"
image = Image.open(imagename).convert("RGB") #requests.get(url, stream=True).raw).convert("RGB")
imageo = cv2.imread(imagename)

reader = easyocr.Reader(['es', 'en'],gpu = False) # load once only in memory.

model = cv2.dnn.readNet('../east/frozen_east_text_detection.pb')

height, width, _ = imageo.shape
dim = (int(width*2), int(height*2))
resized = cv2.resize(imageo, dim, interpolation = cv2.INTER_CUBIC)
height, width, _ = resized.shape

new_height = (height//32)*32
new_width = (width//32)*32
h_ratio = height/new_height
w_ratio = width/new_width
print(h_ratio, w_ratio)
print(new_height, new_width)
img=process(resized)

cv2.imwrite("resized_b.png", resized)

blob = cv2.dnn.blobFromImage(img['contour'], 1, (new_width, new_height),None, True, False) #(123.68, 116.78, 103.94), True, False
model.setInput(blob)
print(model.getUnconnectedOutLayersNames())
(geometry, scores) = model.forward(model.getUnconnectedOutLayersNames())
print((geometry).shape, (scores).shape)

blob2 = cv2.dnn.blobFromImage(img['mask'], 1, (new_width, new_height),None, True, False) #(123.68, 116.78, 103.94), True, False
model.setInput(blob2)
print(model.getUnconnectedOutLayersNames())
(geometry2, scores2) = model.forward(model.getUnconnectedOutLayersNames())
print((geometry2).shape, (scores2).shape)
geometry=geometry+geometry2
scores=scores+scores2
thresh_blurred = cv2.blur(img['canny'], (7, 7))
cv2.imwrite("cthresh_b.png", thresh_blurred)
detected_circles = cv2.HoughCircles(thresh_blurred, cv2.HOUGH_GRADIENT, 2, 30, param1 = 293, param2 = 280, minRadius = 100, maxRadius = 0)
if(not(detected_circles is None)): detected_circles = np.uint16(np.around(detected_circles))
rectangles = [] 
center=(0, 0)
confidence_score = []
for i in range(geometry.shape[2]): 
    for j in range(0, geometry.shape[3]):  
        
        if scores[0][0][i][j] < 0.6:
            continue
        bottom_x = int(j*4 + geometry[0][1][i][j])
        bottom_y = int(i*4 + geometry[0][2][i][j])
        top_x = int(j*4 - geometry[0][3][i][j])
        top_y = int(i*4 - geometry[0][0][i][j])        
        rectangles.append((top_x, top_y, bottom_x, bottom_y))
        confidence_score.append(float(scores[0][0][i][j]))
fin_boxes = non_max_suppression(np.array(rectangles), probs=confidence_score, overlapThresh=0.2)
rectangles=fin_boxes.tolist();

r_easy_ocr=reader.readtext(img['contour'])
for e in r_easy_ocr:
    print(e)
    x1=int(e[0][0][0])
    y1=int(e[0][0][1])
    x2=int(e[0][2][0])
    y2=int(e[0][3][1])
    rectangles.append((x1, y1, x2, y2))

r_easy_ocr=reader.readtext(img['mask'])
for e in r_easy_ocr:
    print(e)
    x1=int(e[0][0][0])
    y1=int(e[0][0][1])
    x2=int(e[0][2][0])
    y2=int(e[0][3][1])
    rectangles.append((x1, y1, x2, y2))

img_copy = resized.copy()
img_copy_t = resized.copy()
if(not(detected_circles is None)):
    for pt in detected_circles[0, :]:
        a, b, r = pt
        print('a:', a, 'b:', b, 'r:', r)
        cv2.circle(img_copy, (a, b), r, (100, 0, 255), 2)
        cv2.circle(img_copy, (a, b), 1, (150, 0, 255), 3)
        cv2.circle(img_copy_t, (a, b), r, (100, 0, 255), 2)
        cv2.circle(img_copy_t, (a, b), 2, (150, 0, 255), 3)
        center=(a, b)
cv2.imwrite("circles.png", img_copy_t)
angles=[]
for (x1, y1, x2, y2) in rectangles:
    ro=[x1,y1,x2,y2]
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0,255,0) , 2) #to draw rectangles on the image
    centerr=[ro[0]+((ro[2]-ro[0])/2), ro[1]+((ro[3]-ro[1])/2)]
    centerr=np.uint16(np.around(centerr))
    cv2.circle(img_copy, centerr, 2, (50, 10, 225), 3)
    delta_x = float(centerr[0]) - float(center[0])
    delta_y = float(centerr[1]) - float(center[1])
    angles.append([ro, round(math.atan2(delta_y, delta_x) * 180 / math.pi,2)])

sorted_angles = sorted(angles, key=lambda x: x[1])
print(sorted_angles)
cl=1
prev=sorted_angles[0][1]
colors = distinctipy.get_colors(len(sorted_angles)+1)
for ([a1, b1, a2, b2], angle) in sorted_angles:
    a1 = int(a1 * w_ratio)
    b1 = int(b1 * h_ratio)
    a2 = int(a2 * w_ratio)
    b2 = int(b2 * h_ratio)
    print (cl, angle, np.multiply(colors[cl], 255))
    if(angle<prev-0.5 or angle>prev+0.5):
        cl=cl+1
    cv2.rectangle(img_copy, (a1, b1), (a2, b2), np.multiply(colors[cl], 255) , 4) #to draw rectangles on the image
    cv2.putText(img_copy, str(angle), (int(a1+((a2-a1)/2)), int(b1+((b2-b1)/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)
    prev=angle


cv2.imwrite("eastdetected.png", img_copy)
print("W")

