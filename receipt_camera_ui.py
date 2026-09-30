"""Desktop interface for capturing receipt photos."""

import base64
import tkinter as tk
from tkinter import messagebox

import cv2

from receipt_camera import ReceiptCamera


def run_camera_ui(camera_index: int = 0, output_dir: str = "receipts") -> None:
    """Show a live camera preview with a button to save a receipt photo."""
    receipt_camera = ReceiptCamera(camera_index=camera_index, output_dir=output_dir)
    root = tk.Tk()
    root.title("Receipt camera")
    root.configure(bg="#17211f")
    root.geometry("900x680")
    root.minsize(640, 480)

    heading = tk.Label(
        root,
        text="Receipt camera",
        font=("Segoe UI", 20, "bold"),
        fg="#f3f4eb",
        bg="#17211f",
    )
    heading.pack(anchor="w", padx=24, pady=(20, 2))
    status = tk.Label(
        root,
        text="Start the camera to frame your receipt.",
        font=("Segoe UI", 10),
        fg="#bdc9c4",
        bg="#17211f",
    )
    status.pack(anchor="w", padx=24, pady=(0, 14))

    preview = tk.Label(root, bg="#0c1211", text="Camera is off", fg="#bdc9c4")
    preview.pack(fill="both", expand=True, padx=24)
    capture_status = tk.Label(
        root,
        text="",
        font=("Segoe UI", 9),
        fg="#b8e0c5",
        bg="#17211f",
    )
    capture_status.pack(anchor="w", padx=24, pady=(8, 0))

    controls = tk.Frame(root, bg="#17211f")
    controls.pack(fill="x", padx=24, pady=18)
    camera_button = tk.Button(
        controls,
        text="Start camera",
        font=("Segoe UI", 10, "bold"),
        bg="#b8e0c5",
        fg="#17211f",
        activebackground="#ccebd5",
        relief="flat",
        padx=16,
        pady=10,
    )
    camera_button.pack(side="left")
    capture_button = tk.Button(
        controls,
        text="Take picture",
        font=("Segoe UI", 10, "bold"),
        bg="#e2a66b",
        fg="#17211f",
        activebackground="#edb982",
        relief="flat",
        padx=16,
        pady=10,
        state="disabled",
    )
    capture_button.pack(side="left", padx=(10, 0))

    camera = None
    latest_frame = None

    def stop_camera() -> None:
        nonlocal camera, latest_frame
        if camera is not None:
            camera.release()
            camera = None
        latest_frame = None
        capture_button.configure(state="disabled")
        camera_button.configure(text="Start camera")
        preview.configure(image="", text="Camera is off")

    def update_preview() -> None:
        nonlocal latest_frame
        if camera is None:
            return

        success, frame = camera.read()
        if not success or frame is None:
            stop_camera()
            status.configure(text="The camera stopped responding.")
            messagebox.showerror("Camera error", "Could not read a frame from the camera.")
            return

        latest_frame = frame
        ready = receipt_camera.check_frame(frame)
        if ready:
            message = "Receipt looks clear and steady"
        else:
            message = "Frame the receipt, focus it, and hold steady"
        status.configure(text=message)

        height, width = frame.shape[:2]
        scale = min(1.0, 840 / width, 500 / height)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        success, encoded_image = cv2.imencode(".png", frame)
        if not success:
            status.configure(text="Could not encode the camera preview.")
            root.after(30, update_preview)
            return
        image_data = base64.b64encode(encoded_image).decode("ascii")
        photo = tk.PhotoImage(data=image_data, format="png")
        preview.configure(image=photo, text="")
        preview.image = photo
        root.after(30, update_preview)

    def toggle_camera() -> None:
        nonlocal camera
        if camera is not None:
            stop_camera()
            status.configure(text="Camera stopped.")
            return

        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            camera.release()
            camera = None
            messagebox.showerror(
                "Camera unavailable",
                "Could not open camera {}. Check camera permissions.".format(camera_index),
            )
            return

        camera_button.configure(text="Stop camera")
        capture_button.configure(state="normal")
        update_preview()

    def capture_photo() -> None:
        if latest_frame is None:
            return
        try:
            photo_path = receipt_camera.save_frame(latest_frame)
        except OSError as error:
            messagebox.showerror("Save failed", str(error))
            return
        capture_status.configure(text="Saved receipt photo: {}".format(photo_path))

    def close_window() -> None:
        stop_camera()
        root.destroy()

    camera_button.configure(command=toggle_camera)
    capture_button.configure(command=capture_photo)
    root.protocol("WM_DELETE_WINDOW", close_window)
    root.mainloop()


if __name__ == "__main__":
    run_camera_ui()