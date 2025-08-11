import cv2
import torch
import datetime
import argparse
import itertools
from pathlib import Path
from rich.live import Live

# Local modules
from modules.process_config import ProcessConfig, create_config
from modules.capture import initialize_video_capture
from modules.saving import SaveConfig, SaveResults
from modules.model_loader import ModelLoader
from modules.annotation import Annotation

# Local tools
from tools.messages import step_message, source_message, progress_table
from tools.general import initialize_display, FPSMonitor


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the video object tracker.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Video Object Tracker")
    parser.add_argument('--source', type=str, required=True, help='Video source (file path or camera index)')
    parser.add_argument('--track', action='store_true', help='Enable tracking mode (default is detection mode)')
    parser.add_argument('--weights', type=str, default='yolo11m.pt', help='Model weights file name')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class ID(s): --classes 0, or --classes 0 2 3')
    parser.add_argument('--size', type=int, default=640, help='Inference size in pixels')
    parser.add_argument('--confidence', type=float, default=0.5, help='Inference confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save output video')

    return parser.parse_args()


def main(config: ProcessConfig) -> None:
    # Initialize process counter
    step_count = itertools.count(1)
    
    # Initialize video capture
    source_info, video_stream = initialize_video_capture(config=config)
    step_message(str(next(step_count)), "Video Source Initialized :white_check_mark:")
    source_message(video_info=source_info)

    # Initialize save results
    saver_config = SaveConfig(
        output_dir=Path(config.source).parent / 'output',
        save_csv=config.save,
        save_video=config.save,
        tracking=config.track )
    results_saver = SaveResults(saver_config)
    results_saver.set_source_info(source_info=source_info)
    step_message(str(next(step_count)), "Saving Results Initialized :white_check_mark:")

    # Check GPU availability
    step_message(str(next(step_count)), f"Processor: {'GPU :white_check_mark:' if torch.cuda.is_available() else 'CPU :warning:'}")

    # Initialize YOLO model
    yolo_tracker = ModelLoader(
        weights_path=config.weights,
        image_size=config.size,
        confidence=config.confidence,
        class_filter=config.classes,
        tracking=config.track )
    step_message(str(next(step_count)), f"{Path(config.weights).stem.upper()} Model Initialized :white_check_mark:")

    # Change live output window size
    window_width, window_height = initialize_display(source_info)

    # Annotators
    annotator = Annotation(
        fps=False,
        trace=True )
    step_message(str(next(step_count)), "Annotations Initialized :white_check_mark:")

    # Variables
    frame_number = 0
    fps_monitor = FPSMonitor()

    # Video processing started
    step_message(str(next(step_count)), "Video Processing Started :white_check_mark:")
    time_start = datetime.datetime.now()
    video_stream.start()
    try:
        with Live(progress_table(frame_number, source_info.total_frames, 0), refresh_per_second=30) as live:
            while video_stream.more() if source_info.source_type == 'file' else True:
                fps_monitor.tick()
                fps_value = fps_monitor.fps()
                
                image = video_stream.read()
                if image is None:
                    print()
                    break

                annotated_image = image.copy()
                
                # Inference
                results = yolo_tracker.inference(image=image)
                    
                # Draw annotations
                for result in results:
                    annotated_image = annotator.on_detections(ultralytics_results=result, scene=annotated_image)

                # Save results if enabled
                if config.save:
                    results_saver.save_csv(filename=source_info.source_name, results=result, frame_number=frame_number)
                    results_saver.save_video(filename=source_info.source_name, image=annotated_image)

                # Presentar progreso en la terminal
                live.update(progress_table(frame_number, source_info.total_frames, fps_value))

                frame_number += 1

                # View live results
                cv2.namedWindow('Resultado', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Resultado', window_width, window_height)
                cv2.imshow("Resultado", annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n")
                    break

    except KeyboardInterrupt:
        step_message(str(next(step_count)), 'End Video ✅')
    
    step_message(str(next(step_count)), f"Total Time: {(datetime.datetime.now() - time_start).total_seconds():.2f} s")
    
    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Carpeta raíz del proyecto
    root_path = Path(__file__).resolve().parent

    options = parse_arguments()
    config = create_config(root_path, options)

    main(config)
