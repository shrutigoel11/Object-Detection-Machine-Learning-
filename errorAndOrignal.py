import pygame
import cv2
import numpy as np
import os

# Initialize the pygame mixer for sound
pygame.mixer.init()

# Load sound files (ensure the paths are correct)
sound_detection = r'C:\Users\91844\Desktop\Object Detection(OpenCv)\sounds\notification-sound-3-262896.mp3'  # New detection sound
sound_error = r'C:\Users\91844\Desktop\Object Detection(OpenCv)\sounds\error.mp3'  # Sound for no object detection

# Check if files exist
if not os.path.exists(sound_detection):
    print(f"Error: {sound_detection} does not exist.")
if not os.path.exists(sound_error):
    print(f"Error: {sound_error} does not exist.")

# Load YOLO model
net = cv2.dnn.readNet('yolo_model\\yolov4.weights', 'yolo_model\\yolov4.cfg')
with open('yolo_model\\yolov3.txt', 'r') as f:
    class_names = f.read().strip().split('\n')

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    detected_objects = False  # Flag for detection

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                detected_objects = True
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label + f" {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Play detection sound if an object is detected
    if detected_objects:
        if not pygame.mixer.music.get_busy():  # Check if sound is already playing
            pygame.mixer.music.load(sound_detection)  # Load the detection sound
            pygame.mixer.music.play()  # Play the sound

    # Play error sound if no object is detected
    else:
        if not pygame.mixer.music.get_busy():  # Check if sound is already playing
            pygame.mixer.music.load(sound_error)  # Load the error sound
            pygame.mixer.music.play()  # Play the sound

    cv2.imshow('frame', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

