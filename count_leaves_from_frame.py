import sys
import os
import cv2
from ultralytics import YOLO

def detect_from_image(cv_image, pt_file):

    # Create model to run
    model = YOLO(pt_file)

    # Run frame through model
    results = model(cv_image)[0]

    detections = results.boxes
    if detections is None:
        return 0, cv_image, detections

    names = results.names
    leaf_mask = []
    for class_id in detections.cls.tolist():
        class_name = str(names[int(class_id)]).lower()
        leaf_mask.append("leaves" in class_name)

    filtered_detections = detections[leaf_mask]
    total = len(filtered_detections)

    # Draw only leaf-tag detections so non-leaf tags are not shown.
    annotated_image = cv_image.copy()
    for xyxy, conf, class_id in zip(
        filtered_detections.xyxy.tolist(),
        filtered_detections.conf.tolist(),
        filtered_detections.cls.tolist(),
    ):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        class_name = str(names[int(class_id)])
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_image,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return total, annotated_image, filtered_detections


if __name__ == '__main__':
    pt = './yolo11n.pt'
    test_file = './canopy1.jpg'
    cv_image = cv2.imread(test_file)
    detect_from_image(cv_image, pt)