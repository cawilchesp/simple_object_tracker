from ultralytics.engine.results import Results

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Any
import cv2
import numpy as np
import pandas as pd

from modules.capture import VideoInfo

@dataclass
class SaveConfig:
    """Configuration for save operations"""
    output_dir: Path
    save_csv: bool = True
    save_video: bool = True
    tracking: bool = False

@dataclass
class FolderPaths:
    """Holds all folder paths for saving different types of data"""
    root: Path
    csv: Optional[Path] = None
    videos: Optional[Path] = None

    @classmethod
    def create(cls, config: SaveConfig) -> 'FolderPaths':
        """Create folder structure based on configuration"""
        root = Path(config.output_dir)
        paths = cls(root=root)

        if config.save_csv:
            paths.csv = root / "resultados" / "csv"
            paths.csv.mkdir(parents=True, exist_ok=True)

        if config.save_video:
            paths.videos = root / "resultados" / "videos"
            paths.videos.mkdir(parents=True, exist_ok=True)

        return paths

class DataWriter(Protocol):
    """Protocol for data writing operations"""
    def write(self, filename: str, data: Any) -> None:
        ...

class VideoWriter:
    """Handles video writing operations"""
    def __init__(self, folder: Path, fps: float, frame_size: tuple):
        self.folder = folder
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.current_file = None

    def write(self, filename: str, frame: np.ndarray) -> None:
        if self.current_file != filename:
            self._create_writer(filename)
        
        self.writer.write(frame)

    def _create_writer(self, filename: str) -> None:
        if self.writer:
            self.writer.release()

        output_path = self.folder / f"{filename}.mp4"
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            self.frame_size
        )
        self.current_file = filename

    def release(self) -> None:
        if self.writer:
            self.writer.release()

class DetectionWriter:
    """Handles detection data writing operations"""
    def __init__(self, folder: Path):
        self.folder = folder

    def write_track(self, filename: str, results: Results, frame_number: int) -> None:
        data = [
            {
                "frame": frame_number,
                "id": int(box.id.item()),
                "class": results.names[int(box.cls)],
                "class_id": int(box.cls),
                "x1": box.xyxy[0][0].item(),
                "y1": box.xyxy[0][1].item(),
                "x2": box.xyxy[0][2].item(),
                "y2": box.xyxy[0][3].item(),
                "conf": box.conf.item()
            }
            for box in results.boxes
        ]
        self._write_dataframe(filename, pd.DataFrame(data))

    def write_detect(self, filename: str, results: Results, frame_number: int) -> None:
        data = [
            {
                "frame": frame_number,
                "class": results.names[int(box.cls)],
                "class_id": int(box.cls),
                "x1": box.xyxy[0][0].item(),
                "y1": box.xyxy[0][1].item(),
                "x2": box.xyxy[0][2].item(),
                "y2": box.xyxy[0][3].item(),
                "conf": box.conf.item()
            }
            for box in results.boxes
        ]
        self._write_dataframe(filename, pd.DataFrame(data))

    def _write_dataframe(self, filename: str, df: pd.DataFrame) -> None:
        path = self.folder / f"{filename}.csv"
        df.to_csv(
            path,
            mode='a' if path.exists() else 'w',
            sep=' ',
            header=False,
            index=False
        )

class SaveResults:
    """Main class for handling all save operations"""
    def __init__(self, config: SaveConfig):
        self.config = config
        self.folders = FolderPaths.create(config)
        self.source_info: Optional[VideoInfo] = None
        
        # Initialize writers
        if config.save_csv:
            self.detection_writer = DetectionWriter(self.folders.csv)

    def set_source_info(self, source_info: VideoInfo) -> None:
        """Set source video information"""
        self.source_info = source_info

        if self.config.save_video:
            self.video_writer = VideoWriter(
                folder=self.folders.videos,
                fps=self.source_info.fps,
                frame_size=(self.source_info.width, self.source_info.height)
            )

    def save_video(self, filename: str, image: np.ndarray) -> None:
        if not self.config.save_video:
            return
        self.video_writer.write(filename, image)

    def save_csv(self, filename: str, results: Results, frame_number: int) -> None:
        if not self.config.save_csv:
            return
        if self.config.tracking:
            self.detection_writer.write_track(filename, results, frame_number)
        else:
            self.detection_writer.write_detect(filename, results, frame_number)
