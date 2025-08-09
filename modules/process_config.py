from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ProcessConfig:
    """Configuration settings for object detection/tracking"""
    source: str
    output: Optional[str]
    track: bool
    weights: Path
    classes: Optional[List[int]]
    size: int
    confidence: float
    csv: bool
    save: bool

def create_config(root_path: Path, args) -> ProcessConfig:
    """Create configuration from command line arguments"""
    return ProcessConfig(
        source=args.source,
        output=args.output,
        track=args.track,
        weights=root_path / "weights" / args.weights,
        classes=args.classes,
        size=args.size,
        confidence=args.confidence,
        csv=args.csv,
        save=args.save
    )