from __future__ import annotations

import cv2
import time
from queue import Queue
from pathlib import Path
from threading import Thread

# Local modules
from modules.process_config import ProcessConfig


class VideoInfo:
    def __init__(
        self,
        source: str
    ) -> None:
        self.source = source

        if self.source.isnumeric():
            self.source_name = "Webcam"
            self.source_type = 'stream'
            video_source = int(self.source)
        elif self.source.lower().startswith('rtsp://'):
            self.source_name = "RSTP Stream"
            self.source_type = 'stream'
            video_source = self.source
        else:
            self.source_name = Path(self.source).stem
            self.source_type = 'file'
            video_source = self.source

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened(): raise IOError('Source video not available ❌')

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source_type == 'file' else None
        

class FileVideoStream:
	def __init__(self, cap, queue_size=128):
		self.stream = cap
		self.stopped = False

		self.Q = Queue(maxsize=queue_size)
		
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True

	def start(self):
		self.thread.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				break

			if not self.Q.full():
				(grabbed, frame) = self.stream.read()
				
				if not grabbed:
					self.stopped = True

				self.Q.put(frame)
			else:
				time.sleep(0.1)

		self.stream.release()

	def read(self):
		return self.Q.get()

	def running(self):
		return self.more() or not self.stopped

	def more(self):
		tries = 0
		while self.Q.qsize() == 0 and not self.stopped and tries < 5:
			time.sleep(0.1)
			tries += 1

		return self.Q.qsize() > 0

	def stop(self):
		self.stopped = True
		self.thread.join()


class WebcamVideoStream:
	def __init__(self, cap, name="WebcamVideoStream"):
		self.stream = cap
		(self.grabbed, self.frame) = self.stream.read()

		self.name = name
		self.stopped = False

	def start(self):
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return

			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True


def initialize_video_capture(
	config: ProcessConfig
) -> tuple[VideoInfo, FileVideoStream | WebcamVideoStream]:
	"""Initialize video capture and return video info and stream"""
	source_info = VideoInfo(source=config.source)
	
	if source_info.source_type == 'stream':
		video_stream = WebcamVideoStream(cap=source_info.cap)
	else:
		video_stream = FileVideoStream(cap=source_info.cap)
	
	return source_info, video_stream
