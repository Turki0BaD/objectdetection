import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_training.cfg') # change to match existing files
classes = []
with open('classes.txt', 'r') as f: # change to match existing files
    classes = f.read().splitlines()

cap = cv2.VideoCapture(1)  # change to select your camera either (0,1,-1) one of the three should do
# img = cv2.imread('office.jpg')

while True:
    ret, img = cap.read()

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.001:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            wid = (x + w) - x
            cov = (wid / 5)  # calibrate the measurments
            cv2.putText(img, str(confidence), (x, y + 20), font, 2, (255, 255, 255), 2)

            # cv2.putText(img, str(confidence) + "cm", (x, y+20), font, 2, (255,255,255), 2)
            if 0 < wid < 75:
                D = 1
            elif 75 < wid < 150:
                D = 2
            elif 150 < wid < 225:
                D = 3
            else:
                D = 4
            print(D)

    cv2.imshow('frame', img)
    key = cv2.waitKey(20)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()