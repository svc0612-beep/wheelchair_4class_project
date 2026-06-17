from ultralytics import YOLO
from pathlib import Path
import multiprocessing


def main():
    # =========================================================
    # 1. 경로 설정
    # =========================================================

    PROJECT_DIR = Path(r"C:\Users\USER\OneDrive\바탕 화면\wheelchair_4class_project")

    DATA_YAML = PROJECT_DIR / "dataset_auto_4class" / "data.yaml"
    RUNS_DIR = PROJECT_DIR / "runs"

    print("data.yaml 경로:", DATA_YAML)
    print("data.yaml 존재:", DATA_YAML.exists())

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            "dataset_auto_4class/data.yaml 파일이 없습니다. "
            "자동 라벨링을 먼저 완료하세요."
        )

    # =========================================================
    # 2. YOLO 모델 로드
    # =========================================================

    model = YOLO("yolov8n.pt")

    # =========================================================
    # 3. 4클래스 YOLO 학습 시작
    # =========================================================

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        patience=15,
        project=str(RUNS_DIR),
        name="wheelchair_4class_yolov8n_50ep",
        workers=0
    )

    print("학습 완료")
    print("결과 저장 위치:")
    print(RUNS_DIR / "wheelchair_4class_yolov8n_50ep")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()