from modules.capture import VideoInfo

from rich import print, box
from rich.table import Table, Column


def step_message(step: str = None, message: str = None) -> None:
    """Display a message with a progress step number.
    Args:
        step (str): The step number or identifier.
        message (str): The message to display.
    """
    print(f"\n[green]\[{step}][/green] {message}")

def source_message(video_info: VideoInfo) -> None:
    """Display video source information in a formatted table.
    Args:
        video_info (VideoInfo): Information about the video source.
    """
    table = Table(
        Column(justify="left", style="bold green"),
        Column(justify="left", style="white", no_wrap=True),
        title="Video Source Information",
        show_header=False,
        box=box.HORIZONTALS )

    table.add_row("Source", f"{video_info.source_name}")
    table.add_row("Size", f"{video_info.width} x {video_info.height}")
    table.add_row("Total Frames", f"{video_info.total_frames}") if video_info.total_frames is not None else None
    table.add_row("Frame Rate", f"{video_info.fps:.2f} FPS")
    
    print()
    print(table)


def progress_table(frame_number: int, total_frames: int, fps_value: float, times: dict = None) -> Table:
    """Create a progress table displaying frame number, FPS, time to end, and optional timing information.
    Args:
        frame_number (int): The current frame number.
        total_frames (int): The total number of frames in the video.
        fps_value (float): The current frames per second value.
        times (dict, optional): A dictionary containing timing information for capture, inference, and total processing times.
    Returns:
        Table: A Rich table object containing the progress information.
    """
    if total_frames is not None:
        percentage = f"[ {frame_number/total_frames:6.1%} ] "
        frame_progress = f"{frame_number} / {total_frames}"
        
        seconds = (total_frames-frame_number) / fps_value  if fps_value != 0 else 0
        hours_process = f"{(seconds // 3600):8.0f}"
        minutes_process = f"{((seconds % 3600) // 60):.0f}"
        seconds_process = f"{(seconds % 60):.2f}"
    else:
        percentage = ''
        frame_progress = f"{frame_number}"
        hours_process = '-'
        minutes_process = '-'
        seconds_process = '-'
    
    table = Table(
        Column(justify="left", style="bold green"),
        Column('Frame', justify="right", style="white", no_wrap=True),
        Column('FPS', justify="right", style="white"),
        Column('Time to End', justify="right", style="white"),
        title="Progress Information",
        box=box.HORIZONTALS )
    
    if times is None:
        table.add_row(f"{percentage}",f"{frame_progress}",f"{fps_value:8.2f}", f"{hours_process}h {minutes_process}m {seconds_process}s")
    else:
        table.add_column('Capture Time', justify="right", style="white")
        table.add_column('Inference Time', justify="right", style="white")
        table.add_column('Frame Time', justify="right", style="white")
        table.add_row(
            f"{percentage}",
            f"{frame_progress}",
            f"{fps_value:8.2f}",
            f"{hours_process}h {minutes_process}m {seconds_process}s",
            f"{1000*(sum(times['capture']) / len(times['capture'])):8.2f} ms",
            f"{1000*(sum(times['inference']) / len(times['inference'])):8.2f} ms",
            f"{1000*(sum(times['total']) / len(times['total'])):8.2f} ms"
        )
    
    return table
