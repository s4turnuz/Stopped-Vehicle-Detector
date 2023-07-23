import argparse
import cv2
import cvzone
import numpy as np
import math
import time
from ultralytics import YOLO
from sort import *

def arguments():

    parser = argparse.ArgumentParser(description="Stopped Vehicle Detection with YOLOv8 and SORT")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to output video file")
    parser.add_argument("--seconds", type=int, required=True, help="Seconds to consider a stopped vehicle")
    parser.add_argument("--yolo_weights", type=str, required=True, help="Path to your yolo model")

    return parser.parse_args()

def definitions(out_file, in_file, weight_file):

    cap = cv2.VideoCapture(in_file)  
    output_file = out_file
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, output_fps, (output_width, output_height))

    model = YOLO(weight_file)
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    detecoes_ids = {}

    return cap, out, model, classNames, tracker, detecoes_ids

def main():
    
    args = arguments()
    cap, out, model, classNames, tracker, detecoes_ids = definitions(args.output, args.input, args.yolo_weights)

    while True:

        success, img = cap.read()

        results = model(img, stream=True)
        detections = np.empty((0, 5))

        imgGraphics = cv2.imread("src/graphics.png", cv2.IMREAD_UNCHANGED)

        height, width, _ = img.shape
        overlay_height, overlay_width, _ = imgGraphics.shape
        pos_x = width - overlay_width

        img = cvzone.overlayPNG(img, imgGraphics, (pos_x, 0))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                    currentArray = tuple([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

            if id not in detecoes_ids.keys():
                detecoes_ids[id] = {
                    'inicio': time.time(),
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                }
            else:
                detecoes_ids[id]['x1'] = x1
                detecoes_ids[id]['y1'] = y1
                detecoes_ids[id]['x2'] = x2
                detecoes_ids[id]['y2'] = y2

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                            scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        ids_parados = []

        for id, detecao in detecoes_ids.items():
            tempo_total = time.time() - detecao['inicio']
            if tempo_total > args.seconds:
                ids_parados.append(id)

        for id in ids_parados:
            detecao = detecoes_ids[id]
            x1, y1, x2, y2 = detecao['x1'], detecao['y1'], detecao['x2'], detecao['y2']
            alpha = 0.3  
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            cv2.putText(img, "STOPPED", (detecao['x1'], detecao['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.putText(img,str(len(ids_parados)),(pos_x+165,100),cv2.FONT_HERSHEY_PLAIN,3,(50,50,255),8)

        ids_remover = []

        for id, detecao in detecoes_ids.items():
            if id not in [int(result[4]) for result in resultsTracker]:
                ids_remover.append(id)

        for id in ids_remover:
            del detecoes_ids[id]

        out.write(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(detecoes_ids)
            break

    out.release()

if __name__ == "__main__":
    main()
