# CS2 Player Detection

An offline computer vision demo that annotates local Counter-Strike 2 screenshots and recordings with object detections.

## Features

- Accepts common local image and video formats.
- Saves bounding boxes, class labels, and confidence scores.
- Supports a configurable confidence threshold, device, and output directory.

## Stack and architecture

`main.py` validates the local files, loads an object detection checkpoint with Ultralytics YOLO, and runs streamed inference. OpenCV writes annotated images as JPEGs. Ultralytics handles annotated video output. Each run gets a separate directory.

## Setup

The available Windows environment uses Python 3.13, PyTorch 2.8.0, Ultralytics 8.3.204, and OpenCV 4.12.0.88.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place a trusted detection checkpoint at `weights/best.pt` or pass an explicit checkpoint path. GPU inference requires a compatible PyTorch build and device.

## Usage

```powershell
.\.venv\Scripts\python.exe main.py inputs/screenshot.png --device cpu
.\.venv\Scripts\python.exe main.py inputs/match.mp4 --weights weights/best.pt --conf 0.35
```

Results go to a unique directory under `outputs/`. Video output is re-encoded without audio.

## Status and provenance

This is an experimental inference demo. Detection quality needs evaluation on labeled footage. The local checkpoint is excluded from Git, and its redistribution terms and the training-data provenance require verification. Code redistribution should retain the upstream attribution and GPL-3.0 license.

Based on the upstream project by TheLlamainator. [CS2-AI-Bot](https://github.com/TheLlamainator/CS2-AI-Bot) · [GPL-3.0](LICENSE)

## Кратко для портфолио

Экспериментальный CLI для разметки заранее записанных изображений и видео через YOLO. Адаптация офлайн-режима опирается на стороннюю разработку. Авторство модели и права на веса требуют подтверждения.
