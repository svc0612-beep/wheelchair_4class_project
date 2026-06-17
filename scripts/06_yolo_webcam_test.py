from ultralytics import YOLO
from pathlib import Path
import cv2
import time

# =========================================================
# 1. 경로 / 설정
# =========================================================

PROJECT_DIR = Path(r"C:\Users\USER\OneDrive\바탕 화면\wheelchair_4class_project")

MODEL_PATH = PROJECT_DIR / "runs" / "wheelchair_4class_yolov8n_50ep-3" / "weights" / "best.pt"

CAMERA_INDEX = 0

# confidence를 너무 낮게 두면 이상한 것도 많이 잡힘
CONF = 0.35

# 화면 속도 제한
TARGET_FPS = 10
FRAME_DELAY = 1.0 / TARGET_FPS

CLASS_NAMES = {
    0: "wheelchair",
    1: "wheelchair_user",
    2: "non_wheelchair_person",
    3: "other_object"
}

print("사용 모델:", MODEL_PATH)
print("모델 존재:", MODEL_PATH.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

# =========================================================
# 2. 모델 로드
# =========================================================

model = YOLO(str(MODEL_PATH))

# =========================================================
# 3. 웹캠 열기
# =========================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    print("CAMERA_INDEX를 1 또는 2로 바꿔보세요.")
    raise SystemExit

print("YOLO 4클래스 웹캠 테스트 시작")
print("q 키를 누르면 종료됩니다.")
print("카메라 번호:", CAMERA_INDEX)

prev_time = time.time()
fps = 0.0

# =========================================================
# 4. 실시간 추론
# =========================================================

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

    results = model.predict(
        source=frame,
        conf=CONF,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    # 화면 상단 정보 표시
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f} | CONF: {CONF}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "0 wheelchair | 1 wheelchair_user | 2 non_wheelchair_person | 3 other_object",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # 탐지 결과를 터미널에 너무 많이 찍지 않고, 화면에만 표시
    cv2.imshow("Wheelchair 4Class YOLO Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # FPS 제한
    loop_time = time.time() - loop_start
    sleep_time = FRAME_DELAY - loop_time

    if sleep_time > 0:
        time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()

print("YOLO 웹캠 테스트 종료")