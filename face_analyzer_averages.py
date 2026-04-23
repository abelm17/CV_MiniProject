import cv2
import mediapipe as mp
from scipy.spatial import distance
from mediapipe.tasks.python import vision

import urllib.request
from pathlib import Path

MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

if not Path(MODEL_PATH).exists():
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

FACE_TOP, FACE_BOTTOM = 10, 152
FACE_LEFT, FACE_RIGHT = 234, 454
NOSE_TOP, NOSE_TIP    = 168, 4
LEFT_NOSTRIL, RIGHT_NOSTRIL = 129, 358
MOUTH_LEFT, MOUTH_RIGHT     = 61, 291
MOUTH_TOP, MOUTH_BOTTOM     = 13, 14

REF_IPD_MM  = 63.0
FRAME_SKIP  = 4  # process every Nth frame

def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C) if C else 0

def get_pt(landmarks, idx, w, h):
    return (landmarks[idx].x * w, landmarks[idx].y * h)

def px_to_mm(px, ipd_px):
    return px * (REF_IPD_MM / ipd_px) if ipd_px else 0

def estimate_dimensions(landmarks, w, h):
    pt = lambda i: get_pt(landmarks, i, w, h)
    dist = distance.euclidean

    left_pupil  = ((pt(33)[0] + pt(133)[0]) / 2, (pt(33)[1] + pt(133)[1]) / 2)
    right_pupil = ((pt(362)[0] + pt(263)[0]) / 2, (pt(362)[1] + pt(263)[1]) / 2)
    ipd_px = dist(left_pupil, right_pupil)
    mm = lambda px: px_to_mm(px, ipd_px)

    return {
        "face_height_mm":  mm(dist(pt(FACE_TOP),     pt(FACE_BOTTOM))),
        "face_width_mm":   mm(dist(pt(FACE_LEFT),    pt(FACE_RIGHT))),
        "left_eye_w_mm":   mm(dist(pt(133),          pt(33))),
        "right_eye_w_mm":  mm(dist(pt(362),          pt(263))),
        "nose_height_mm":  mm(dist(pt(NOSE_TOP),     pt(NOSE_TIP))),
        "nose_width_mm":   mm(dist(pt(LEFT_NOSTRIL), pt(RIGHT_NOSTRIL))),
        "mouth_width_mm":  mm(dist(pt(MOUTH_LEFT),   pt(MOUTH_RIGHT))),
        "mouth_height_mm": mm(dist(pt(MOUTH_TOP),    pt(MOUTH_BOTTOM))),
    }

def analyze_video(video_path):
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker_v2_with_blendshapes.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    EAR_THRESH   = 0.21
    CONSEC_MIN   = 3
    counter      = 0
    total_blinks = 0
    frame_num    = 0
    dim_accum    = {}
    dim_samples  = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # skip frames without decoding for speed
        if frame_num % FRAME_SKIP != 0:
            continue

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect_for_video(mp_image, int(frame_num * 1000 / fps))

        if not result.face_landmarks:
            continue

        lms = result.face_landmarks[0]
        left_eye  = [(int(lms[i].x * w), int(lms[i].y * h)) for i in LEFT_EYE_IDX]
        right_eye = [(int(lms[i].x * w), int(lms[i].y * h)) for i in RIGHT_EYE_IDX]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        if ear < EAR_THRESH:
            counter += 1
        else:
            if counter >= CONSEC_MIN:
                total_blinks += 1
            counter = 0

        if frame_num % 15 == 0:
            dims = estimate_dimensions(lms, w, h)
            for k, v in dims.items():
                dim_accum[k] = dim_accum.get(k, 0) + v
            dim_samples += 1

    cap.release()

    avg_dims = {k: v / dim_samples for k, v in dim_accum.items()} if dim_samples else {}
    return {
        "total_blinks":   total_blinks,
        "blinks_per_min": total_blinks / (duration_sec / 60),
        "blinks_per_sec": total_blinks / duration_sec,
        "duration_sec":   duration_sec,
        "dimensions":     avg_dims,
    }

video_folder = Path("videos")
all_results = []
for video_path in sorted(video_folder.glob("*.[Mm][Pp]4")):
    print(f"Processing {video_path.name}...")
    result = analyze_video(str(video_path))
    all_results.append(result)

total_blinks   = sum(r["total_blinks"] for r in all_results)
total_duration = sum(r["duration_sec"] for r in all_results)

print(f"\n══ COMBINED SUMMARY ({len(all_results)} videos) ══")
print(f"  Total duration : {total_duration:.0f}s ({total_duration/3600:.2f}h)")
print(f"  Total blinks   : {total_blinks}")
print(f"  Blinks/min     : {total_blinks / (total_duration / 60):.4f}")
print(f"  Blinks/sec     : {total_blinks / total_duration:.6f}")

all_dims = [r["dimensions"] for r in all_results if r["dimensions"]]
if all_dims:
    avg_dims = {k: sum(d[k] for d in all_dims) / len(all_dims) for k in all_dims[0]}
    print(f"  Face H x W     : {avg_dims['face_height_mm']:.1f} x {avg_dims['face_width_mm']:.1f} mm")
    print(f"  Left eye W     : {avg_dims['left_eye_w_mm']:.1f} mm  |  Right eye W: {avg_dims['right_eye_w_mm']:.1f} mm")
    print(f"  Nose H x W     : {avg_dims['nose_height_mm']:.1f} x {avg_dims['nose_width_mm']:.1f} mm")
    print(f"  Mouth W x H    : {avg_dims['mouth_width_mm']:.1f} x {avg_dims['mouth_height_mm']:.1f} mm")