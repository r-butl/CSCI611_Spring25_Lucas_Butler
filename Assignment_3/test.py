import cv2
from ultralytics import YOLO
import os

# Load trained model
model = YOLO("/home/lrbutler/Desktop/611/Assignment_3/runs/detect/train6/weights//best.pt")  # Update path if needed

# Define threshold ranges for optimization
confidence_thresholds = [0.9]
nms_thresholds = [0.9]

# Open video file
input_video_path = "test_video.small.mp4"  # Update with your video file path
output_video_path = "/research2/lrbutler"

# Get video properties
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Iterate through threshold combinations
for conf_thresh in confidence_thresholds:
    for nms_thresh in nms_thresholds:
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Define output video writer with unique filename
        output_filename = os.path.join(output_video_path, f"output_conf{conf_thresh}_nms{nms_thresh}.mp4")
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no more frames

            # Run inference with current thresholds
            results = model(frame, conf=conf_thresh, iou=nms_thresh)

            # Process results
            for result in results:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    if conf > conf_thresh:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write frame to output video
            out.write(frame)

        # Release video writer for the current configuration
        out.release()

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Inference completed. Optimized videos saved.")
