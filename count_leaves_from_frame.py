import sys
import os
#Ensure virtual environment is used for ultralytics import
venv_path ='/opt/pyenv/lib/python3.12/site-packages'
if venv_path not in sys.path:
    sys.pat.insert(0,venv_path)

from ultralytics import YOLO

def detect_from_image(cv_image, pt_file):

    # Create model to run
    model = YOLO(pt_file)

    # Run frame through model
    results = model(cv_image)[0]

    detections = results.boxes
    total = 0 if detections is None else len(detections)
    annotated_image = results.plot()

    return total, annotated_image, detections


if __name__ == '__main__':
    pt = './yolo11n.pt'
    test_file = './canopy1.jpg'
    cv_image = cv2.imread(test_file)
    detect_from_image(cv_image, pt)