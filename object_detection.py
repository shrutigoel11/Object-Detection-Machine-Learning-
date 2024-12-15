import cv2
import numpy as np
import pygame  # For playing sounds

# Initialize the pygame mixer for sound
pygame.mixer.init()

# Load MP3 sound file (make sure the path is correct for your system)
pygame.mixer.music.load('detection_sound.mp3')  # Change this to your MP3 file path

# Load YOLO
net = cv2.dnn.readNet('yolo_model\\yolov4.weights', 'yolo_model\\yolov4.cfg')

# Load the class names from yolov3.txt
with open('yolo_model\\yolov3.txt', 'r') as f:
    class_names = f.read().strip().split('\n')

# Get the layers names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the output
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = False  # Flag to track if an object was detected

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                detected_objects = True  # Set flag to True if any object is detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Ensure that indices is not None and flatten it
    if len(indices) > 0:
        indices = indices.flatten()

        for i in indices:
            x, y, w, h = boxes[i]
            label = str(class_names[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label + f" {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Play sound if objects were detected
    if detected_objects:
        if not pygame.mixer.music.get_busy():  # Check if music is already playing
            pygame.mixer.music.play()

    # Show the output
    cv2.imshow('frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
