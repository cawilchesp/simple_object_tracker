from modules.capture import VideoInfo
from pathlib import Path

from rich.table import Table, Column
from rich import print, box


from icecream import ic

# Constants
# ---------
FG_RED = '\033[31m'
FG_GREEN = '\033[32m'
FG_YELLOW = '\033[33m'
FG_BLUE = '\033[34m'
FG_WHITE = '\033[37m'
FG_BOLD = '\033[01m'
FG_RESET = '\033[0m'

# Funciones de color
# ------------------
def bold(text: str) -> str:
    return f"{FG_BOLD}{text}{FG_RESET}" if text is not None else ''

def red(text: str) -> str:
    return f"{FG_RED}{text}{FG_RESET}" if text is not None else ''

def green(text: str) -> str:
    return f"{FG_GREEN}{text}{FG_RESET}" if text is not None else ''

def yellow(text: str) -> str:
    return f"{FG_YELLOW}{text}{FG_RESET}" if text is not None else ''

def blue(text: str) -> str:
    return f"{FG_BLUE}{text}{FG_RESET}" if text is not None else ''

def white(text: str) -> str:
    return f"{FG_WHITE}{text}{FG_RESET}" if text is not None else ''

# Funciones
# ---------
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


def progress_message(frame_number: int, total_frames: int, fps_value: float):
    if total_frames is not None:
        percentage_title = f"{'':11}"
        percentage = f"[ {frame_number/total_frames:6.1%} ] "
        frame_progress = f"{frame_number} / {total_frames}"
        
        seconds = (total_frames-frame_number) / fps_value  if fps_value != 0 else 0
        hours_process = f"{(seconds // 3600):8.0f}"
        minutes_process = f"{((seconds % 3600) // 60):.0f}"
    else:
        percentage_title = ''
        percentage = ''
        frame_progress = f"{frame_number}"
        hours_process = '        -'
        minutes_process = '-'
    
    frame_text_length = (2 * len(str(total_frames))) + 3
    if frame_number == 0:
        print(f"\n{percentage_title}{bold('Frame'):>{frame_text_length+9}}{bold('FPS'):>22}{bold('Est. End (h)'):>27}")
    print(f"\r{green(percentage)}{frame_progress:>{frame_text_length}}     {fps_value:8.2f}     {hours_process}h {minutes_process}m  ", end="", flush=True)
    

def times_message(frame_number: int, total_frames: int, fps_value: float, times: dict):
    if total_frames is not None:
        percentage_title = f"{'':11}"
        percentage = f"[ {frame_number/total_frames:6.1%} ] "
        frame_progress = f"{frame_number} / {total_frames}"
        
        seconds = (total_frames-frame_number) / fps_value  if fps_value != 0 else 0
        hours_process = seconds // 3600
        minutes_process = (seconds % 3600) // 60
    else:
        percentage_title = ''
        percentage = ''
        frame_progress = f"{frame_number}"
        hours_process = ''
        minutes_process = ''
        
    if frame_number == 0:
        print(
            f"\n{percentage_title}"
            f"{bold('Frame'):>{(2 * len(str(total_frames))) + 12}}"
            f"{bold('Capture'):>22}"
            f"{bold('Inference'):>22}"
            f"{bold('Total'):>22}"
            f"{bold('FPS'):>22}"
            f"{bold('Est. End (h)'):>27}" )
    print(
        f"\r{green(percentage)}"
        f"{frame_progress:>{(2 * len(str(total_frames))) + 3}}  "
        f"{1000*(sum(times['capture']) / len(times['capture'])):8.2f} ms  "
        f"{1000*(sum(times['inference']) / len(times['inference'])):8.2f} ms  "
        f"{1000*(sum(times['total']) / len(times['total'])):8.2f} ms     "
        f"{fps_value:8.2f}     "
        f"{hours_process:8.0f}h {minutes_process:.0f}m  ",
        end="", flush=True )
