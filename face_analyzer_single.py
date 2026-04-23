import cv2
import mediapipe as mp
from scipy.spatial import distance
from mediapipe.tasks.python import vision

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Landmarks for dimension estimation
FACE_TOP, FACE_BOTTOM = 10, 152
FACE_LEFT, FACE_RIGHT = 234, 454
NOSE_TOP, NOSE_TIP    = 168, 4
LEFT_NOSTRIL, RIGHT_NOSTRIL = 129, 358
MOUTH_LEFT, MOUTH_RIGHT     = 61, 291
MOUTH_TOP, MOUTH_BOTTOM     = 13, 14

# Average adult IPD used as real-world scale reference (mm)
REF_IPD_MM = 63.0

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

    # IPD from inner/outer eye corners as scale reference
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

        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect_for_video(mp_image, int(frame_num * 1000 / fps))
        frame_num += 1

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

        # accumulate dimensions every 5 frames
        if frame_num % 5 == 0:
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

def print_report(label, r):
    print(f"\n── {label} ──")
    print(f"  Duration       : {r['duration_sec']:.0f}s ({r['duration_sec']/3600:.2f}h)")
    print(f"  Total blinks   : {r['total_blinks']}")
    print(f"  Blinks/min     : {r['blinks_per_min']:.4f}")
    print(f"  Blinks/sec     : {r['blinks_per_sec']:.6f}")
    d = r["dimensions"]
    if d:
        print(f"  Face H x W     : {d['face_height_mm']:.1f} x {d['face_width_mm']:.1f} mm")
        print(f"  Left eye W     : {d['left_eye_w_mm']:.1f} mm  |  Right eye W: {d['right_eye_w_mm']:.1f} mm")
        print(f"  Nose H x W     : {d['nose_height_mm']:.1f} x {d['nose_width_mm']:.1f} mm")
        print(f"  Mouth W x H    : {d['mouth_width_mm']:.1f} x {d['mouth_height_mm']:.1f} mm")

# RENAME VIDEO TO CHANGE VIDEOS HERE
video= "../videos/GX011037.MP4"

result = analyze_video(video)
print_report("Session Results:", result)