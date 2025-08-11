from ultralytics.engine.results import Results

import cv2
import numpy as np
from collections import defaultdict, deque

from tools.general import FPSMonitor


class Trace:
    """Class to store and manage object tracking points.
    Attributes:
        xy (defaultdict): Dictionary to store tracking points for each object.
    Args:
        max_size (int): Maximum number of points to store for each object.
    """
    def __init__(
        self,
        max_size: int = 30,
    ) -> None:
        self.xy = defaultdict(lambda: deque(maxlen=max_size))

    def add_points(self, xyxys: np.ndarray, tracker_ids: np.ndarray) -> None:
        """Adds tracking points for each object.
        Args:
            xyxys (np.ndarray): Bounding box coordinates of the objects.
            tracker_ids (np.ndarray): Tracking IDs of the objects.
        """
        for xyxy, tracker_id in zip(xyxys, tracker_ids):
            track_id = int(tracker_id)
            x1, y1, x2, y2 = xyxy
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.xy[track_id].append((center_x, center_y))

    def get_points(self, track_id: int) -> np.ndarray:
        """Returns the tracking points for a specific object.
        Args:
            track_id (int): Tracking ID of the object.
        Returns:
            np.ndarray: Array of tracking points for the object.
        """
        return np.array(self.xy[track_id]).astype(np.int32) if track_id in self.xy else None


class KeyPoints:
    """Class to store key points of detected objects.
    Args:
        xy (np.ndarray): Bounding box coordinates of the objects.
        class_id (np.ndarray, optional): Class IDs of the objects.
        confidence (np.ndarray, optional): Confidence scores of the detections.
        tracker_id (np.ndarray, optional): Tracking IDs of the objects.
        class_names (np.ndarray, optional): Names of the classes.
    Attributes:
        xy (np.ndarray): Bounding box coordinates of the objects.
        class_id (np.ndarray): Class IDs of the objects.
        confidence (np.ndarray): Confidence scores of the detections.
        tracker_id (np.ndarray): Tracking IDs of the objects.
        class_names (np.ndarray): Names of the classes.
    """
    def __init__(
        self,
        xy: np.ndarray,
        class_id: np.ndarray = None,
        confidence: np.ndarray = None,
        tracker_id: np.ndarray = None,
        class_names: np.ndarray = None
    ) -> None:
        self.xy = xy
        self.class_id = class_id
        self.confidence = confidence
        self.tracker_id = tracker_id
        self.class_names = class_names


class Annotation:
    """Clase para la anotación de objetos en un video.
    Args:
        fps (bool): Mostrar velocidad de procesamiento en fotogramas por segundo.
        label (bool): Mostrar etiquetas de las detecciones.
        box (bool): Mostrar cajas delimitadoras de las detecciones.
        trace (bool): Mostrar rastros de seguimiento de las detecciones.
        colorbox (bool): Mostrar cajas con colores de las detecciones.
        heatmap (bool): Mostrar mapas de calor.
        mask (bool): Mostrar máscaras de segmentación.
        track_length (int): Longitud de los rastros de seguimiento.
        color_opacity (float): Opacidad de las cajas de colores.
    """
    def __init__(
        self,
        fps: bool = True,
        label: bool = True,
        box: bool = True,
        trace: bool = False,
        colorbox: bool = False,
        heatmap: bool = False,
        mask: bool = False,
        track_length: int = 50,
        color_opacity: float = 0.5,
    ) -> None:
        self.fps = fps
        self.label = label
        self.box = box
        self.trace = trace
        self.colorbox = colorbox
        self.heatmap = heatmap
        self.mask = mask
        self.color_opacity = color_opacity
        
        self.COLOR_LIST = [
            (0,204,255),  # amarillo
            (7,127,15),   # verde
            (255,149,0),  # azul
            (85,45,255),  # rojo
            (0,149,255),  # naranja
            (240,240,70), # cian
            (222,82,207), # morado
            (60,245,210)  # verde limon
        ]

        if self.fps: self.fps_monitor = FPSMonitor()
        if self.trace: self.trajectories = Trace(max_size=track_length)
        if self.heatmap: self.heat_mask = None


    def on_detections(self, ultralytics_results: Results, scene: np.ndarray) -> np.ndarray:
        """Draws annotations on the input image.
        Args:
            ultralytics_results (Results): Inference results from Ultralytics.
            scene (np.array): Input image.
        Returns:
            np.array: Image with annotations.
        """
        detections = self._extract_detections(ultralytics_results=ultralytics_results)
        
        if self.fps:
            scene = self._draw_fps(scene=scene)
        
        if self.box:
            scene = self.box_annotator(
                scene=scene,
                xyxys=detections['xyxys'],
                class_ids=detections['class_ids'])
        
        if self.label:
            object_labels = self._generate_labels(detections=detections)
            scene = self.label_annotator(
                scene=scene,
                xyxys=detections['xyxys'],
                class_ids=detections['class_ids'],
                labels=object_labels )
        
        if self.trace and detections['tracker_ids'] is not None:
            scene = self.trace_annotator(
                scene=scene,
                xyxys=detections['xyxys'],
                class_ids=detections['class_ids'],
                tracker_ids=detections['tracker_ids'] )
        
        if self.colorbox:
            scene = self.colorbox_annotator(
                scene=scene,
                xyxys=detections['xyxys'],
                class_ids=detections['class_ids'] )
        
        if self.heatmap:
            scene = self.heatmap_annotator(
                scene=scene,
                xyxys=detections['xyxys'] )
        
        return scene

    def _extract_detections(self, ultralytics_results: Results) -> dict:
        """Extracts detection data from Ultralytics results.
        Args:
            ultralytics_results (Results): Inference results from Ultralytics.
        Returns:
            dict: Dictionary containing detection data.
        """
        detections = {
            'xyxys': ultralytics_results.boxes.xyxy.cpu().numpy(),
            'mask': ultralytics_results.masks.data.cpu().numpy() if ultralytics_results.masks is not None else None,
            'confidence': ultralytics_results.boxes.conf.cpu().numpy(),
            'class_ids': ultralytics_results.boxes.cls.cpu().numpy().astype(int),
            'tracker_ids': ultralytics_results.boxes.id.int().cpu().numpy() if ultralytics_results.boxes.id is not None else None,
            'class_names': np.array([ultralytics_results.names[i] for i in ultralytics_results.boxes.cls.cpu().numpy().astype(int)])
        }
        return detections

    def _draw_fps(self, scene: np.ndarray) -> np.ndarray:
        """Draws FPS on the input image.
        Args:
            scene (np.array): Input image.
        Returns:
            np.array: Image with FPS.
        """
        self.fps_monitor.tick()
        fps_value = self.fps_monitor.fps()
        return self.draw_text(scene, f"{fps_value:.1f} FPS")

    def _generate_labels(self, detections: dict) -> list:
        """Generates labels for detections.
        Args:
            detections (dict): Dictionary containing detection data.
        Returns:
            list: List of labels.
        """
        if detections['tracker_ids'] is None:
            return [f"{class_name} ({confidence:.2f})"
                    for class_name, confidence
                    in zip(detections['class_names'], detections['confidence'])]
        else:
            return [f"{class_name} {tracker_id} ({confidence:.2f})"
                    for class_name, tracker_id, confidence
                    in zip(detections['class_names'], detections['tracker_ids'], detections['confidence'])]

    def draw_text(self, scene: np.ndarray, text: str) -> np.ndarray:
        """Dibuja el texto en la imagen de entrada.
        Args:
            text (str): Texto a dibujar.
            scene (np.array): Imagen de entrada.
        Returns:
            np.array: Imagen con texto.
        """
        text_scale = 0.6
        text_thickness = 1

        (text_w, text_h) = cv2.getTextSize(
            text=text,
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=text_scale,
            thickness=text_thickness )[0]

        cv2.rectangle(
            img=scene,
            pt1=(10, 40 - text_h - 8),
            pt2=(10+text_w+4, 40),
            color=(0,0,0),
            thickness=-1 )
        
        cv2.putText(
            img=scene,
            text=text,
            org=(10 + 2, 40 - 4),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=text_scale,
            color=(255,255,255),
            thickness=text_thickness,
            lineType=cv2.LINE_AA )

        return scene


    def box_annotator(self, scene: np.ndarray, xyxys: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """Draws bounding boxes on the input image.
        Args:
            scene (np.array): Input image.
            xyxys (np.ndarray): Bounding box coordinates of the objects.
            class_ids (np.ndarray): Class IDs of the objects.
        Returns:
            np.array: Image with bounding boxes.
        """
        for xyxy, class_id in zip(xyxys, class_ids):
            x1, y1, x2, y2 = xyxy.astype(int)
            color = self.COLOR_LIST[class_id.astype(int) % len(self.COLOR_LIST)]
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color,
                thickness=1 )

        return scene
    
    
    def label_annotator(self, scene: np.ndarray, xyxys: np.ndarray, class_ids: np.ndarray, labels: list) -> np.ndarray:
        """Draws labels on the input image.
        Args:
            scene (np.array): Input image.
            xyxys (np.ndarray): Bounding box coordinates of the objects.
            class_ids (np.ndarray): Class IDs of the objects.
            labels (list): List of labels to draw.
        Returns:
            np.array: Image with labels.
        """
        text_scale = 0.4
        text_thickness = 1
        for xyxy, class_id, label in zip(xyxys, class_ids, labels):
            (text_w, text_h) = cv2.getTextSize(
                text=label,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=text_scale,
                thickness=text_thickness )[0]

            x1, y1, _, _ = xyxy.astype(int)
            color = self.COLOR_LIST[class_id.astype(int) % len(self.COLOR_LIST)]
            cv2.rectangle(
                img=scene,
                pt1=(x1, y1 - text_h - 8),
                pt2=(x1+text_w+4, y1),
                color=color,
                thickness=-1 )
            cv2.putText(
                img=scene,
                text=label,
                org=(x1 + 2, y1 - 4),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=text_scale,
                color=(0,0,0),
                thickness=text_thickness,
                lineType=cv2.LINE_AA )

        return scene
   
   
    def trace_annotator(self, scene: np.ndarray, xyxys: np.ndarray, class_ids: np.ndarray, tracker_ids: np.ndarray) -> np.ndarray:
        """Draws tracking traces on the input image.
        Args:
            scene (np.array): Input image.
            xyxys (np.ndarray): Bounding box coordinates of the objects.
            class_ids (np.ndarray): Class IDs of the objects.
            tracker_ids (np.ndarray): Tracking IDs of the objects.
        Returns:
            np.array: Image with tracking traces.
        """
        self.trajectories.add_points(xyxys, tracker_ids)
        line_thickness = 1
        for xyxy, class_id, tracker_id in zip(xyxys, class_ids, tracker_ids):
            track_id = int(tracker_id)
            color = self.COLOR_LIST[class_id.astype(int) % len(self.COLOR_LIST)]
            xy = self.trajectories.get_points(track_id=track_id)
            if len(xy) > 1:
                scene = cv2.polylines(
                    img=scene,
                    pts=[xy],
                    isClosed=False,
                    color=color,
                    thickness=line_thickness )

        return scene


    def colorbox_annotator(self, scene: np.ndarray, xyxys: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """Draws colored boxes on the input image.
        Args:
            scene (np.array): Input image.
            xyxys (np.ndarray): Bounding box coordinates of the objects.
            class_ids (np.ndarray): Class IDs of the objects.
        Returns:
            np.array: Image with colored boxes.
        """
        scene_with_boxes = scene.copy()
        for xyxy, class_id in zip(xyxys, class_ids):
            x1, y1, x2, y2 = xyxy.astype(int)
            color = self.COLOR_LIST[class_id.astype(int) % len(self.COLOR_LIST)]
            cv2.rectangle(
                img=scene_with_boxes,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color,
                thickness=-1 )

        cv2.addWeighted(
            src1=scene_with_boxes,
            src2=scene,
            alpha=self.color_opacity,
            beta=1 - self.color_opacity,
            gamma=0,
            dst=scene )

        return scene

    def heatmap_annotator(self, scene: np.ndarray, xyxys: np.ndarray) -> np.ndarray:
        """Draws a heatmap on the input image.
        Args:
            scene (np.array): Input image.
            xyxys (np.ndarray): Bounding box coordinates of the objects.
        Returns:
            np.array: Image with heatmap.
        """
        if self.heat_mask is None:
            self.heat_mask = np.zeros(scene.shape[:2], dtype=np.float32)
        
        mask = np.zeros(scene.shape[:2])
        for x1, y1, x2, y2 in xyxys:
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            cv2.circle(
                img=mask,
                center=(int(center_x), int(center_y)),
                radius=40,
                color=(1,),
                thickness=-1 )

        self.heat_mask = mask + self.heat_mask

        temp = self.heat_mask.copy()
        temp = 125 - temp / temp.max() * (125 - 0)
        temp = temp.astype(np.uint8)
        temp = cv2.blur(temp, (25,25))
        hsv = np.zeros(scene.shape)
        hsv[..., 0] = temp
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        mask = cv2.cvtColor(self.heat_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR) > 0

        scene[mask] = cv2.addWeighted(temp, 0.2, scene, 1 - 0.2, 0)[mask]

        return scene
    