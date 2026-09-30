# Receipt camera

`ReceiptCamera` opens a laptop camera preview, looks for a receipt-shaped
rectangle, checks that the receipt is in focus and steady, and saves a JPEG
automatically when those conditions are met.

## Install

```bash
python -m pip install -r requirements.txt
```

## Use

Launch the desktop camera interface:

```bash
python receipt_camera_ui.py
```

Choose **Start camera**, frame the receipt, then choose **Take picture**. Photos
are saved in the `receipts` folder.

You can also use the camera from Python:

```python
from receipt_camera import ReceiptCamera

camera = ReceiptCamera(output_dir="receipts")
photo_path = camera.capture_when_ready(timeout_seconds=60)
if photo_path is not None:
    print("Saved receipt photo to {}".format(photo_path))
```

Keep the full receipt visible, well lit, and steady in the preview. Press `q`
to cancel. If no clear, steady receipt is found before the optional timeout,
`TimeoutError` is raised. Camera and image-save errors are also reported to the
caller.

The default camera is device `0`. Pass another `camera_index` if the laptop
camera is exposed under a different device number. Tune `sharpness_threshold`,
`min_receipt_area`, and `stable_frames` when constructing `ReceiptCamera` if
your camera or lighting needs different detection settings.
