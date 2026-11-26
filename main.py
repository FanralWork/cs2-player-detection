"""Run YOLOv8 player detection on a local Counter-Strike 2 image or video."""

import argparse
from pathlib import Path
from tempfile import mkdtemp


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "best.pt"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local image or video")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Local YOLO .pt file")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold (default: 0.2)")
    parser.add_argument("--device", help="Inference device, such as cpu or 0 (default: automatic)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Results root (default: outputs)")
    args = parser.parse_args()

    source = args.source.expanduser().absolute()
    suffix = source.suffix.lower()
    if not source.is_file() or suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        parser.error(f"expected a local image or video file: {source}")

    weights = args.weights.expanduser().absolute()
    if not weights.is_file() or weights.suffix != ".pt":
        parser.error(f"expected a local .pt file at {weights}; see the README for model setup")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")

    import cv2
    from ultralytics import YOLO

    is_video = suffix in VIDEO_SUFFIXES
    try:
        model = YOLO(str(weights))
        if model.task != "detect":
            parser.error("the weights must be for an object detection model")

        output_root = args.output_dir.expanduser().absolute()
        output_root.mkdir(parents=True, exist_ok=True)
        save_dir = Path(mkdtemp(prefix="predict-", dir=output_root))
        results = model.predict(
            source=str(source),
            conf=args.conf,
            imgsz=640,
            device=args.device,
            project=str(output_root),
            name=save_dir.name,
            exist_ok=True,
            save=is_video,
            stream=True,
            verbose=False,
        )
        processed = False
        for result in results:
            processed = True
            if not is_video:
                encoded, image = cv2.imencode(".jpg", result.plot())
                if not encoded:
                    raise OSError("could not encode the annotated image")
                (save_dir / f"{source.stem}.jpg").write_bytes(image.tobytes())

        if not processed:
            raise OSError("could not read any frames from the input file")
        if not any(file.is_file() and file.stat().st_size for file in save_dir.iterdir()):
            raise OSError(f"could not write the annotated output in {save_dir}")
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Detection failed: {error}\n")

    print(f"Saved annotated output to {save_dir}")


if __name__ == "__main__":
    main()
