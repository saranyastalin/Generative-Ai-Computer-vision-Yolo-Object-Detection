from ultralytics import YOLO
import cv2
import cvzone

# Load YOLO model (nano = fastest)
model = YOLO("yolov10n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Reduce camera resolution (IMPORTANT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

frame_count = 0
skip_frames = 2   # YOLO runs every 2 frames

while True:
    ret, image = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO only every N frames
    if frame_count % skip_frames == 0:
        results = model(image, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0]) * 100
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                detections.append((x1, y1, x2, y2, class_name, confidence))

    # Draw last detections (smooth display)
    if 'detections' in locals():
        for x1, y1, x2, y2, class_name, confidence in detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(
                image,
                f"{class_name} {confidence:.1f}%",
                (x1, y1 - 10),
                scale=1,
                thickness=2
            )

    cv2.imshow("YOLO Detection", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
