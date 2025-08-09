from collections import defaultdict, deque
from datetime import datetime

class ProcessTimer:
    """Class to measure and store processing times for different stages.
    Attributes:
        times (defaultdict): A dictionary to store lists of time measurements for each stage.
    Args:
        max_samples (int): Maximum number of samples to keep for each stage.
    """
    def __init__(self, max_samples=30):
        self.times = defaultdict(lambda: deque(maxlen=max_samples))
        
    def add_measurement(self, name: str, start_time: datetime, end_time: datetime) -> None:
        """Add a time measurement for a specific process
        Args:
            name (str): The name of the process stage.
            start_time (datetime): The start time of the measurement.
            end_time (datetime): The end time of the measurement.
        """
        duration = (end_time - start_time).total_seconds()
        self.times[name].append(duration)
    