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
