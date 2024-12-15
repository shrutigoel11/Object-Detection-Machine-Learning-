# Object Detection with YOLOv4 and OpenCV

This repository demonstrates real-time object detection using the **YOLOv4 (You Only Look Once)** model with **OpenCV** for video capture and processing. The project captures live video from your webcam, uses YOLOv4 to detect various objects, and provides sound notifications when objects are detected or when no objects are found.

## Features:
- **Real-Time Object Detection**: Detects multiple objects in live video using the YOLOv4 deep learning model.
- **Bounding Boxes**: Draws bounding boxes around detected objects in each video frame.
- **Sound Alerts**: Plays a custom sound when an object is detected, and a different sound when no object is detected.
- **Customizable Detection**: Easily change detection settings and replace sound files for different alerts.

## Requirements:
- **Python 3.x**
- **OpenCV**: `opencv-python`
- **Pygame**: `pygame` (for sound handling)
- **YOLOv4 Model Files**:
  - Pre-trained **YOLOv4 weights**
  - YOLOv4 **configuration file** (`yolov4.cfg`)
  - **Class names file** (`coco.names`)

## Setup Instructions:
1. Clone the repository:
    ```bash
    git clone https://github.com/shrutigoel11/Object-Detection-Machine-Learning-.git
    ```

2. Install the required Python libraries:
    ```bash
    pip install opencv-contrib-python cvlib gtts PyObjC pygame numpy tensorflow==2.10.0
    ```

3. Download the YOLOv4 model files:
   - **YOLOv4 weights**: [Download here](https://github.com/AlexeyAB/darknet/releases)
   - **YOLOv4 configuration file**: [Download here](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)
   - **Class names file**: [Download here](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names)

4. Place the downloaded files in the `yolo_model/` folder in the project directory.

5. Run the object detection script:
    ```bash
    python object_detection.py
    ```

## How It Works:
- Captures video from the webcam.
- Processes each frame using YOLOv4 to identify objects.
- Draws bounding boxes around detected objects.
- Plays a notification sound if an object is detected, or an error sound if no objects are detected.

## Use Cases:
- **Surveillance Systems**: Automatically detect and track objects in security camera feeds.
- **AI Assistants**: Enable devices to recognize objects and respond accordingly.
- **Industrial Automation**: Automate inspections and monitoring in production lines.

## Contributing:
Feel free to fork this repository, open issues, and submit pull requests. Contributions are welcome!
