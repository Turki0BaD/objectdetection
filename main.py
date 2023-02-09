import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg') #dnn= deep neural network. this line input the trainer module
classes = [] # The objects for project ours is only 1.
with open('classes.txt', 'r') as f: # Inputing classes from .txt file
    classes = f.read().splitlines()
# for classes we don't typaclly use a file for 1 class. However since the project turned from general detection to singular detection. we used a file to keep things simple
cap = cv2.VideoCapture(0)  # This line allows for you to choose a camera (0,1,-1,2) are all valiables


while True: #open loop for video
    ret, img = cap.read() # read footage from camera

    height, width, _ = img.shape # the shape of the video

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False) #turn the footage into a format the deep learning can understand
    net.setInput(blob) # set dnn to work inside the blob
    output_layers_names = net.getUnconnectedOutLayersNames() #getout put from yolov3 layers (82,94,106) if i'm not mistaken
    layerOutputs = net.forward(output_layers_names) #make the dnn work with the layers
# the 2 previous lines are a bit diffuclt to understand, but they are responisble for the layers yolov3 use and dnn thinks of it as a way to use get an output from trained modules
    boxes = [] # initaiat a variable this one is to make the squares around the nuts
    confidences = [] # initaiat a variable this one is responsible for how sure yolov3 that an object is in the class
    class_ids = [] # initaiat a variable this one is responsible for every class within the classes.txt file but we only have 1

    for output in layerOutputs: # if object is detected enters this for loop
        for detection in output: # 
            scores = detection[5:] # get class id
            class_id = np.argmax(scores) # also getting the class id
            confidence = scores[class_id] # get the confidance of the detected object... example (0% sure to 100% sure)
            if confidence > 0.001: # if loop to decide if the object detected is rated high enough to be outputted... example(if confidate enough declare as a Nut) 
                center_x = int(detection[0] * width) #the lines from 32 all the way to 42 are responsible of assigning the cooardnates for the box then (40) display the box.
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2) # declare detected object

    font = cv2.FONT_HERSHEY_PLAIN # not really important just font change
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3)) #same as before set randome color to the box

    if len(indexes) > 0: # if indexes declares an object at least enter the if loop
        for i in indexes.flatten(): #make frame readable for dnn
            x, y, w, h = boxes[i] # call the box from line 51 to 54
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            wid = (x + w) - x # detected object width
            cov = (wid / 5)  # calibrate the measurments
            cv2.putText(img, str(confidence), (x, y + 20), font, 2, (255, 255, 255), 2) # output text along sidethe box in this case (output confidance perceantage)

            # cv2.putText(img, str(confidence) + "cm", (x, y+20), font, 2, (255,255,255), 2) # output text along sidethe box in this case (output confidance perceantage) can replace with the width or the cov parameter
            if 0 < wid < 75: #start a loop to decide the direction of the Nut... size calibration
                D = 1
            elif 75 < wid < 150:
                D = 2
            elif 150 < wid < 225:
                D = 3
            else:
                D = 4
            print(D)
# from the previous loop D is our output for the size of the Nut
    cv2.imshow('frame', img) #show footage
    key = cv2.waitKey(20) # responsiable for assigning a key to dismiss the system
    if key == 27: #ESC to shut down the system
        break #exit loop

cap.release() #shutdown the footage
cv2.destroyAllWindows() # remove all windows and related to the code
