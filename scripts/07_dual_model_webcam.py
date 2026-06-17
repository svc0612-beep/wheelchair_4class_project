from ultralytics import YOLO
from pathlib import Path
import cv2
import time


# =========================================================
# 1. path / settings
# =========================================================

PROJECT_DIR = Path(r"C:\Users\USER\OneDrive\바탕 화면\wheelchair_4class_project")

CUSTOM_MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "wheelchair_4class_yolov8n_50ep-3"
    / "weights"
    / "best.pt"
)
BASE_MODEL_PATH = PROJECT_DIR / "yolov8n.pt"

# 05_camera_test.py에서 1번 카메라가 성공했으므로 1번을 먼저 시도한다.
CAMERA_INDEXES = [1, 0, 2]

# 모델 호출용 최소값. 실제 표시는 아래 클래스별 threshold를 다시 적용한다.
CUSTOM_CONF = 0.25
BASE_CONF = 0.45

CUSTOM_CLASS_CONF = {
    "wheelchair": 0.45,
    "wheelchair_user": 0.65,
}

CHAIR_GUARD_LABELS = {"chair", "bench", "couch"}
CHAIR_GUARD_OVERLAP = 0.35
WHEELCHAIR_USER_CHAIR_CONF = 0.93

TARGET_FPS = 10
FRAME_DELAY = 1.0 / TARGET_FPS

CUSTOM_CLASS_NAMES = {
    0: "wheelchair",
    1: "wheelchair_user",
}

CUSTOM_COLORS = {
    "wheelchair": (0, 220, 255),
    "wheelchair_user": (0, 80, 255),
}
BASE_COLOR = (80, 255, 80)

IOU_SUPPRESS_THRESHOLD = 0.35


# =========================================================
# 2. helpers
# =========================================================

def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def box_overlap_over_smaller(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return 0.0

    return inter_area / smaller_area


def to_detections(result, class_names=None, allowed_classes=None, source_name="base"):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])

        if allowed_classes is not None and class_id not in allowed_classes:
            continue

        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        if class_names is None:
            label = result.names.get(class_id, str(class_id))
        else:
            label = class_names.get(class_id, str(class_id))

        min_conf = CUSTOM_CLASS_CONF.get(label)

        if min_conf is not None and conf < min_conf:
            continue

        detections.append(
            {
                "box": xyxy,
                "conf": conf,
                "label": label,
                "source": source_name,
            }
        )

    return detections


def suppress_chair_like_wheelchair_users(custom_detections, base_detections):
    filtered = []
    chair_like_detections = [
        det for det in base_detections if det["label"] in CHAIR_GUARD_LABELS
    ]

    for custom_det in custom_detections:
        if custom_det["label"] != "wheelchair_user":
            filtered.append(custom_det)
            continue

        overlaps_chair = any(
            box_overlap_over_smaller(custom_det["box"], chair_det["box"])
            >= CHAIR_GUARD_OVERLAP
            for chair_det in chair_like_detections
        )

        if overlaps_chair and custom_det["conf"] < WHEELCHAIR_USER_CHAIR_CONF:
            continue

        filtered.append(custom_det)

    return filtered


def remove_overlapped_base_detections(custom_detections, base_detections):
    filtered = []

    for base_det in base_detections:
        should_hide = False

        for custom_det in custom_detections:
            if box_iou(base_det["box"], custom_det["box"]) >= IOU_SUPPRESS_THRESHOLD:
                should_hide = True
                break

        if not should_hide:
            filtered.append(base_det)

    return filtered


def draw_detection(frame, detection):
    x1, y1, x2, y2 = [int(v) for v in detection["box"]]
    label = detection["label"]
    conf = detection["conf"]

    if detection["source"] == "custom":
        color = CUSTOM_COLORS.get(label, (0, 220, 255))
        text = f"CUSTOM {label} {conf:.2f}"
    else:
        color = BASE_COLOR
        text = f"{label} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_w, text_h = text_size
    label_y = max(0, y1 - text_h - baseline - 6)

    cv2.rectangle(
        frame,
        (x1, label_y),
        (x1 + text_w + 8, label_y + text_h + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 4, label_y + text_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
    )


# =========================================================
# 3. model load
# =========================================================

print("커스텀 모델:", CUSTOM_MODEL_PATH)
print("기본 YOLO 모델:", BASE_MODEL_PATH)
print("커스텀 모델 존재:", CUSTOM_MODEL_PATH.exists())
print("기본 YOLO 모델 존재:", BASE_MODEL_PATH.exists())

if not CUSTOM_MODEL_PATH.exists():
    raise FileNotFoundError(f"커스텀 모델 파일을 찾을 수 없습니다: {CUSTOM_MODEL_PATH}")

if not BASE_MODEL_PATH.exists():
    raise FileNotFoundError(f"기본 YOLO 모델 파일을 찾을 수 없습니다: {BASE_MODEL_PATH}")

custom_model = YOLO(str(CUSTOM_MODEL_PATH))
base_model = YOLO(str(BASE_MODEL_PATH))


# =========================================================
# 4. webcam open
# =========================================================

def open_camera():
    for index in CAMERA_INDEXES:
        print(f"카메라 {index}번 확인 중...")
        test_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not test_cap.isOpened():
            test_cap.release()
            continue

        valid_frame = False

        for _ in range(10):
            ret, frame = test_cap.read()

            if ret and frame is not None and frame.size > 0 and frame.mean() > 1:
                valid_frame = True
                break

            time.sleep(0.05)

        if valid_frame:
            return test_cap, index

        test_cap.release()

    return None, None


cap, camera_index = open_camera()

if cap is None:
    print("웹캠을 열 수 없습니다.")
    print("다른 Python/OpenCV 창이 켜져 있으면 먼저 종료해보세요.")
    raise SystemExit

print("듀얼 모델 웹캠 테스트 시작")
print("CUSTOM: wheelchair / wheelchair_user")
print("BASE YOLO: COCO 80 classes")
print("OpenCV 화면을 한 번 클릭한 뒤 q 또는 ESC 키를 누르면 종료됩니다.")
print("카메라 번호:", camera_index)

prev_time = time.time()
fps = 0.0


# =========================================================
# 5. realtime inference
# =========================================================

try:
    while True:
        loop_start = time.time()

        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽지 못했습니다.")
            break

        now = time.time()
        elapsed = now - prev_time

        if elapsed > 0:
            fps = 1 / elapsed

        prev_time = now

        custom_result = custom_model.predict(
            source=frame,
            conf=CUSTOM_CONF,
            verbose=False,
        )[0]

        base_result = base_model.predict(
            source=frame,
            conf=BASE_CONF,
            verbose=False,
        )[0]

        custom_detections = to_detections(
            custom_result,
            class_names=CUSTOM_CLASS_NAMES,
            allowed_classes={0, 1},
            source_name="custom",
        )
        base_detections = to_detections(base_result, source_name="base")

        custom_detections = suppress_chair_like_wheelchair_users(
            custom_detections,
            base_detections,
        )

        base_detections = remove_overlapped_base_detections(
            custom_detections,
            base_detections,
        )

        display_frame = frame.copy()

        for detection in base_detections:
            draw_detection(display_frame, detection)

        for detection in custom_detections:
            draw_detection(display_frame, detection)

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f} | CUSTOM_CONF: {CUSTOM_CONF} | BASE_CONF: {BASE_CONF}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "CUSTOM: wheelchair / wheelchair_user | BASE YOLO: COCO classes | q/ESC: quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Dual Model YOLO Webcam", display_frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("q") or key == 27:
            break

        loop_time = time.time() - loop_start
        sleep_time = FRAME_DELAY - loop_time

        if sleep_time > 0:
            time.sleep(sleep_time)
finally:
    cap.release()
    cv2.destroyAllWindows()

print("듀얼 모델 웹캠 테스트 종료")
