# simple_object_tracker

**simple_object_tracker** is a Python-based tool and a base project for object detection and tracking using YOLO models. It is designed to be modular, easy to use, and extensible for various computer vision tasks.

## Features

- Object detection using YOLOv1 models (pretrained weights included)
- Modular architecture for annotation, model loading, and result saving
- Configurable processing pipeline
- Tools for timing, messaging, and general utilities

## Project Structure

- main.py # Entry point for running the tracker
- modules/
    - annotation.py # Annotation utilities 
    - capture.py # Video/image capture logic 
    - model_loader.py # Model loading and inference 
    - process_config.py # Configuration processing 
    - saving.py # Saving results and outputs 
- tools/
    - general.py # General utilities 
    - messages.py # Messaging/logging helpers 
    - timing.py # Timing and performance tools 

## Usage

- Place your input videos or images in the appropriate directory.
- Run `main.py` to start detection and tracking.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Carlos Andrés Wilches Pérez