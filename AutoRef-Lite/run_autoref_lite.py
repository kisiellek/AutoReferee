import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from geometry import point_side_of_line

BALL_CLS = "sports ball"  # COCO model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="out/overlay.mp4")
    p.add_argument("--csv", default="out/events.csv")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--conf", type=float, default=0.75)
    p.add_argument("--smooth", type=int, default=3, help="okno średniej kroczącej")
    p.add_argument("--cooldown", type=int, default=20, help="klatki blokady po golu")
    return p.parse_args()


def pick_goal_line(frame):
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))

    clone = frame.copy()
    cv2.namedWindow("Pick goal line: 2 clicks")
    cv2.setMouseCallback("Pick goal line: 2 clicks", on_mouse)
    while True:
        disp = clone.copy()
        for p in pts:
            cv2.circle(disp, p, 5, (0, 255, 255), -1)
        if len(pts) == 2:
            cv2.line(disp, pts[0], pts[1], (0, 255, 255), 2)
        cv2.putText(
            disp,
            "Click two points for goal line, then press ENTER",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Pick goal line: 2 clicks", disp)
        key = cv2.waitKey(20)
        if key in (13, 10) and len(pts) == 2:  # ENTER
            break
        if key == 27:  # ESC
            raise SystemExit("Aborted by user")
    cv2.destroyWindow("Pick goal line: 2 clicks")
    return tuple(pts[0]), tuple(pts[1])


def main():
    args = parse_args()
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    # Pierwsza klatka do kalibracji linii
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Empty video")

    line_p1, line_p2 = pick_goal_line(first)

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.out), fourcc, cap.get(cv2.CAP_PROP_FPS) or 30.0, (w, h)
    )

    model = YOLO(args.model)

    # bufor wygładzający centroid piłki
    centroids = deque(maxlen=args.smooth)

    # logika gola
    side_prev = None
    goals = 0
    cooldown = 0

    events = []

    def draw_line(img):
        cv2.line(img, line_p1, line_p2, (0, 255, 255), 2)

    # wróć na początek wideo
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # detekcja
        res = model.predict(frame, conf=args.conf, verbose=False)[0]

        ball_bbox = None
        ball_conf = -1.0
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            name = res.names[cls_id]
            if name == BALL_CLS:
                conf = float(box.conf[0].item())
                if conf > ball_conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    ball_bbox = (x1, y1, x2, y2)
                    ball_conf = conf

        if ball_bbox is not None:
            x1, y1, x2, y2 = ball_bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            centroids.append((cx, cy))

            sm_cx = int(np.mean([c[0] for c in centroids]))
            sm_cy = int(np.mean([c[1] for c in centroids]))

            # rysowanie
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(frame, (sm_cx, sm_cy), 5, (0, 200, 0), -1)

            # wektor prędkości (px/frame)
            if len(centroids) >= 2:
                vx = centroids[-1][0] - centroids[-2][0]
                vy = centroids[-1][1] - centroids[-2][1]
                cv2.arrowedLine(
                    frame, (sm_cx, sm_cy), (sm_cx + vx * 2, sm_cy + vy * 2), (0, 200, 0), 2
                )

            # wykrycie gola przez zmianę strony względem linii
            side_now = point_side_of_line((sm_cx, sm_cy), line_p1, line_p2)
            if side_prev is None:
                side_prev = side_now

            if cooldown > 0:
                cooldown -= 1
            else:
                if side_prev is not None and side_now is not None and side_now != side_prev:
                    goals += 1
                    cooldown = args.cooldown
                    events.append(
                        {
                            "frame": frame_idx,
                            "time_s": (time.time() - t0),
                            "event": "GOAL",
                            "cx": sm_cx,
                            "cy": sm_cy,
                            "conf": ball_conf,
                        }
                    )
                side_prev = side_now
        else:
            # brak piłki w tej klatce
            centroids.clear()

        # HUD
        draw_line(frame)
        cv2.putText(
            frame,
            f"Goals: {goals}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        cv2.imshow("AutoRef Lite", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # zapis CSV
    df = pd.DataFrame(events)
    if not df.empty:
        df.to_csv(args.csv, index=False)
        print(f"Saved events -> {args.csv}")
    else:
        print("No events detected. CSV skipped.")


if __name__ == "__main__":
    main()

