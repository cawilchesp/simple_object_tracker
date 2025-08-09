import torch
from ultralytics import YOLO

def main():
    print("Tracker env check:")
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.is_available())
    # Example: load a small model to prove ultralytics works (no training)
    try:
        YOLO("yolov8n.pt")
        print("Ultralytics loaded a YOLO model successfully.")
    except Exception as e:
        print("Ultralytics model load failed:", e)

if __name__ == "__main__":
    main()
