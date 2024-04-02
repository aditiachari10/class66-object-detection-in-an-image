import cv2
import numpy as np  # np is numerical python to handle the array and matrix data
labels = open("coco.names").read().strip().split('\n')
print(labels)
# Path to model configuration and weights files
modelConfiguration='cfg/yolov3.cfg'

# Load YOLO object detection network
modelWeights='yolov3.weights'

# Load image
yoloNetwork= cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

image=cv2.imread('static/img1.jpg')
# Get image dimensions
image= cv2.resize(image,(800,800))
#image=image.shape
#print(image)

dimension=image.shape[:2]
print(dimension)
H=dimension[0]
W=dimension[1]

boxes=[]
confidences=[]
classIds=[]
thresholdConfidence= 0.5
NMSThreshold= 0.3
# Create blob from image and set input for YOLO network
blob=cv2.dnn.blobFromImage(image, 1/255,(416, 416))
yoloNetwork.setInput(blob)

# Create unconnected layers
layerName= yoloNetwork.getUnconnectedOutLayersNames()

# Pass the unconnected layes to the yolo network
layerOutputs = yoloNetwork.forward(layerName)
#print(layerName)
#print(layerOutputs)
for output in layerOutputs:
    for detection in output:
        scores=detection[5:]
        classId=np.argmax(scores)
        confidence=scores[classId]

        if confidence > thresholdConfidence:
            box=detection[0:4]*np.array([W,H,W,H])
            
            (centerX, centerY, width, height) = box.astype('int')
            x= int(centerX-(width/2))
            y= int(centerY-(height/2))
       
            boxes.append([x,y,int(width),int(height)])
            confidences.append(confidence)
            classIds.append(classId)
print(boxes)
print(confidences)
print(classIds)

indexes=cv2.dnn.NMSBoxes(boxes, confidences, thresholdConfidence, NMSThreshold)

for i in range(len(boxes)):
    if i in indexes: #check if individual boxes are present 
        x=boxes[i][0]
        y=boxes[i][1]
        w=boxes[i][2]
        h=boxes[i][3]

        label = labels[classIds[i]]
        text='{}:{:.2f}'.format(label, confidences[i]*100)
        cv2.putText(image, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)

        cv2.rectangle(image,(x,y), (x+w, y+h), (0,0,255), 2)
# Syntax: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size)
# 1/255 is takes to normalise the pixel value farom 0-255 to 0-1 as the yolo (other models also) require the pixel to be in range 0 to 1.
# 416,416 is size of images taken by yolo model


# Display image
cv2.imshow('Image', image)
cv2.waitKey(0)