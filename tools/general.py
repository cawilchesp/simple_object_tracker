import time
from collections import deque

from modules.capture import VideoInfo


def initialize_display(source_info: VideoInfo, noshow: bool = True) -> tuple[int, int]:
    """Initialize display window settings
    Args:
        source_info (VideoInfo): Information about the video source.
        noshow (bool): If True, do not show the display window.
    Returns:
        tuple[int, int]: The width and height of the display window.
    """
    if not noshow:
        scaled_width = 1280 if source_info.width > 1280 else source_info.width
        scaled_height = int(scaled_width * source_info.height / source_info.width)
        scaled_height = scaled_height if source_info.height > scaled_height else source_info.height
        
        return int(scaled_width), int(scaled_height)
    
    return int(source_info.width), int(source_info.height)


class FPSMonitor:
    """Class to monitor frames per second (FPS) performance.
    This class uses a deque to store timestamps of frame captures and calculates the
    FPS based on these timestamps.
    Attributes:
        timestamps (deque): A deque to store timestamps of frame captures.
    Args:
        sample_size (int): The maximum number of timestamps to store.
    Example:
        fps_monitor = FPSMonitor(sample_size=30)
        fps_monitor.tick()
        fps_value = fps_monitor.fps()
    """
    def __init__(self, sample_size: int = 30):
        self.timestamps = deque(maxlen=sample_size)

    def fps(self) -> float:
        """Calculate the current frames per second (FPS).
        Returns:
            float: The calculated FPS value.
        """
        if not self.timestamps:
            return 0.0
        elapsed_time = self.timestamps[-1] - self.timestamps[0]
        
        return (len(self.timestamps)) / elapsed_time if elapsed_time != 0 else 0.0

    def tick(self) -> None:
        """Record the current timestamp."""
        self.timestamps.append(time.monotonic())