from ultralytics import YOLO
import cv2 
import math
import torch
import time

prev_time = 0

cap = cv2.VideoCapture("./videos/cars.mp4") # for videos

# Check PyTorch device
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))



model = YOLO("./Yolo-Weights/yolo11l.pt")
print(model.names)

allowed_classes = ["car", "truck", "bus", "motorcycle", "motorbike", "bicycle"] 

# READS the mask
mask = cv2.imread("mask.png")

#line cords
limits = [400, 297, 673, 297]

counted_ids = set()
total_count = 0

while True:
  success, img = cap.read()

  # creates mask region from video
  imgRegion = cv2.bitwise_and(img, mask)


  results = model.track(imgRegion, tracker="bytetrack.yaml", stream=True, persist=True)
  for r in results:
    boxes = r.boxes
    for box in boxes:
      
      # confidence
      conf = math.ceil((box.conf[0] * 100))/100
      
      # class names
      cls = int(box.cls[0]) 
    
      class_name = model.names[cls]

      # bounding box
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

      box_h = y2 - y1
      font_scale = max(0.55, min(2.0, box_h / 120))

      if class_name in allowed_classes and conf>0.30:

        # print(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        track_id = int(box.id[0]) if box.id is not None else None

        label = f"{class_name} {conf}"
        if track_id is not None:
          label = f"{label} ID:{track_id}"

        # display conf and class
        cv2.putText(img, label, (max(0, x1), max(30, y1-10)),
              cv2.FONT_HERSHEY_SIMPLEX,
              font_scale, (255, 0, 255), 2)
        
      # line which dictates the tracking
      cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
      
      # circles that when go over line, will count the car (visualization)
      w, h = x2-x1, y2-y1
      cx, cy = x1+w//2, y1+h//2
      cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

      if limits[0] <cx< limits[2] and limits[1]-20 < cy < limits[1]+20:
        if track_id not in counted_ids:
          counted_ids.add(track_id)
          total_count +=1
          cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

      # TEXT CONTENT
      count_text = f"Count: {total_count}"

      # text size
      (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

      # BOX BACKGROUND
      cv2.rectangle(img, (40, 20), (40 + text_w + 20, 20 + text_h + 20 + text_h + 20), (0, 0, 0), -1)

      # TEXT
      cv2.putText(img, count_text, (50, 15 + text_h + 15),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
      
    # --- FPS CALCULATION ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(img, f"FPS: {int(fps)}", (50, 15 + text_h*3 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


  #shows image
  cv2.imshow('Image', img)
  # cv2.imshow('ImageRegion', imgRegion)
  cv2.waitKey(1)

  # python car_counter.py 