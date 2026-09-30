"""Capture a receipt from a laptop camera once it is in focus and steady."""

from collections import deque
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Deque, Optional, Tuple

import cv2
import numpy as np


class ReceiptCamera:
    """Find a receipt in the camera preview and save it when it is clear.

    A receipt is considered ready when a rectangular receipt outline is visible,
    its image sharpness passes the configured threshold, and its position is
    steady for several consecutive frames.
    """

    def __init__(
        self,
        camera_index: int = 0,
        output_dir: str = "receipts",
        sharpness_threshold: float = 100.0,
        min_receipt_area: float = 0.12,
        stable_frames: int = 5,
    ) -> None:
        if sharpness_threshold < 0:
            raise ValueError("sharpness_threshold must be non-negative")
        if not 0 < min_receipt_area < 1:
            raise ValueError("min_receipt_area must be between 0 and 1")
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")

        self.camera_index = camera_index
        self.output_dir = Path(output_dir)
        self.sharpness_threshold = sharpness_threshold
        self.min_receipt_area = min_receipt_area
        self.stable_frames = stable_frames
        self._recent_positions: Deque[Tuple[float, float, float]] = deque(
            maxlen=stable_frames
        )
        self._receipt_outline: Optional[np.ndarray] = None
        self._sharpness = 0.0
        self._receipt_detected = False

    def check_frame(self, frame: np.ndarray) -> bool:
        """Return whether this frame completes a clear, steady receipt view."""
        self._receipt_outline = self._find_receipt(frame)
        self._receipt_detected = self._receipt_outline is not None
        if self._receipt_outline is None:
            self._recent_positions.clear()
            self._sharpness = 0.0
            return False

        self._sharpness = self._measure_sharpness(frame, self._receipt_outline)
        if self._sharpness < self.sharpness_threshold:
            self._recent_positions.clear()
            return False

        height, width = frame.shape[:2]
        moments = cv2.moments(self._receipt_outline)
        if moments["m00"] == 0:
            self._recent_positions.clear()
            return False

        center_x = moments["m10"] / moments["m00"] / width
        center_y = moments["m01"] / moments["m00"] / height
        area = cv2.contourArea(self._receipt_outline) / float(width * height)
        self._recent_positions.append((center_x, center_y, area))

        if len(self._recent_positions) < self.stable_frames:
            return False

        positions = np.asarray(self._recent_positions)
        center_spread = np.sqrt(
            (positions[:, 0] - positions[:, 0].mean()) ** 2
            + (positions[:, 1] - positions[:, 1].mean()) ** 2
        ).max()
        area_mean = positions[:, 2].mean()
        area_spread = np.abs(positions[:, 2] - area_mean).max() / area_mean
        return center_spread <= 0.02 and area_spread <= 0.15

    def capture_when_ready(
        self,
        show_preview: bool = True,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[Path]:
        """Open the camera and save one photo when the receipt is ready.

        Press ``q`` in the preview to cancel. A timeout raises ``TimeoutError``;
        camera access, frame-read, and image-write errors raise ``RuntimeError``
        or ``OSError`` with details.
        """
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")

        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError(
                "Could not open camera {}. Check the camera index and permissions."
                .format(self.camera_index)
            )

        started_at = monotonic()
        window_name = "Receipt camera"
        window_created = False
        try:
            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    raise RuntimeError("The camera opened but could not read a frame.")

                ready = self.check_frame(frame)
                if show_preview:
                    self._show_status(frame, ready, window_name)
                    window_created = True
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return None

                if ready:
                    return self._save_frame(frame)

                if (
                    timeout_seconds is not None
                    and monotonic() - started_at >= timeout_seconds
                ):
                    raise TimeoutError(
                        "No clear, steady receipt was detected before the timeout."
                    )
        finally:
            camera.release()
            if window_created:
                cv2.destroyWindow(window_name)

    def _find_receipt(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        frame_area = float(frame.shape[0] * frame.shape[1])

        best_outline = None
        best_area = 0.0
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            outline = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(outline) != 4 or not cv2.isContourConvex(outline):
                continue

            area = cv2.contourArea(outline)
            if area / frame_area < self.min_receipt_area or area <= best_area:
                continue
            best_outline = outline
            best_area = area

        return best_outline

    @staticmethod
    def _measure_sharpness(frame: np.ndarray, outline: np.ndarray) -> float:
        points = outline.reshape(4, 2).astype("float32")
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(4)
        ordered = np.zeros((4, 2), dtype="float32")
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(differences)]
        ordered[3] = points[np.argmax(differences)]

        width = int(
            max(
                np.linalg.norm(ordered[2] - ordered[3]),
                np.linalg.norm(ordered[1] - ordered[0]),
            )
        )
        height = int(
            max(
                np.linalg.norm(ordered[1] - ordered[2]),
                np.linalg.norm(ordered[3] - ordered[0]),
            )
        )
        if width < 2 or height < 2:
            return 0.0

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype="float32",
        )
        transform = cv2.getPerspectiveTransform(ordered, destination)
        receipt = cv2.warpPerspective(frame, transform, (width, height))
        receipt_gray = cv2.cvtColor(receipt, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(receipt_gray, cv2.CV_64F).var())

    def _save_frame(self, frame: np.ndarray) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = "receipt_{}.jpg".format(
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        photo_path = self.output_dir / filename
        if not cv2.imwrite(str(photo_path), frame):
            raise OSError("Could not save receipt photo to {}".format(photo_path))
        return photo_path

    def save_frame(self, frame: np.ndarray) -> Path:
        """Save a camera frame as a receipt photo."""
        return self._save_frame(frame)

    def _show_status(self, frame: np.ndarray, ready: bool, window_name: str) -> None:
        preview = frame.copy()
        if self._receipt_outline is not None:
            cv2.polylines(preview, [self._receipt_outline], True, (0, 220, 0), 2)

        if ready:
            message = "Receipt clear and steady - capturing"
            color = (0, 220, 0)
        elif not self._receipt_detected:
            message = "Move the full receipt into the frame"
            color = (0, 165, 255)
        elif self._sharpness < self.sharpness_threshold:
            message = "Hold still and improve focus"
            color = (0, 165, 255)
        else:
            message = "Keep the receipt steady"
            color = (0, 165, 255)

        cv2.putText(
            preview,
            "{} (sharpness: {:.0f})".format(message, self._sharpness),
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.putText(
            preview,
            "Press q to cancel",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.imshow(window_name, preview)
