from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ProcessConfig:
    """Configuration settings for object detection/tracking
    Attributes:
        source (str): Video source (file path or camera index)
        track (bool): Enable tracking mode (default is detection mode)
        weights (Path): Path to the model weights file
        classes (Optional[List[int]]): Filter by class ID(s)
        size (int): Inference size in pixels
        confidence (float): Inference confidence threshold
        save (bool): Save output video
    """
    source: str
    track: bool
    weights: Path
    classes: Optional[List[int]]
    size: int
    confidence: float
    save: bool

def create_config(root_path: Path, args) -> ProcessConfig:
    """Create configuration from command line arguments
    Args:
        root_path (Path): Root path of the project
        args (argparse.Namespace): Parsed command line arguments
    """
    return ProcessConfig(
        source=args.source,
        track=args.track,
        weights=root_path / "weights" / args.weights,
        classes=args.classes,
        size=args.size,
        confidence=args.confidence,
        save=args.save,
    )