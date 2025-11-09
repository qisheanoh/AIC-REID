from pathlib import Path
import cv2
import numpy as np
from .bot_sort import BOTSORT


def main():
    video_path = Path("data/raw/videos/terrace1-c0.avi")
    output_path = Path("runs/terrace_out.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path.resolve()}")
        return

    print(f"[INFO] Opening video: {video_path.resolve()}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[ERROR] Cannot open video file")
        return

    # Get frame properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0 or np.isnan(fps):
        print("[WARN] FPS is zero, using fallback 30 FPS.")
        fps = 30.0
    print(f"[DEBUG] Frame size: {width}x{height}, FPS: {fps}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    print(f"[DEBUG] VideoWriter opened? {out.isOpened()}")
    if not out.isOpened():
        print("[ERROR] Failed to open video writer!")
        cap.release()
        return
    print(f"[INFO] Output video will be saved to: {output_path.resolve()}")

    # Initialize tracker
    tracker = BOTSORT(device="cpu")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or read error")
            break

        print(f"[INFO] Processing frame {frame_id}")
        results = tracker.update(frame, frame_id)

        for x1, y1, x2, y2, track_id in results:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show and save
        cv2.imshow("BoT-SORT + ReID", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing complete.")


if __name__ == "__main__":
    main()
