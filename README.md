# Wheelchair 4-Class YOLO Project

YOLOv8 기반의 휠체어 관련 객체 탐지 프로젝트입니다.  
휠체어, 휠체어 사용자, 일반 보행자, 기타 객체를 4개 클래스로 구분해 학습하고 웹캠으로 테스트하는 흐름을 포함합니다.

## Classes

이 프로젝트는 다음 4개 클래스를 사용합니다.

| ID | Class | Description |
| --- | --- | --- |
| 0 | `wheelchair` | 사람이 타고 있지 않은 휠체어 |
| 1 | `wheelchair_user` | 휠체어에 탄 사람 |
| 2 | `non_wheelchair_person` | 휠체어를 사용하지 않는 사람 |
| 3 | `other_object` | 휠체어와 혼동될 수 있는 기타 객체 |

자세한 라벨링 기준은 `LABELING_GUIDE_4CLASS.txt`를 참고하세요.

## Repository Structure

```text
.
├── dataset/
│   └── data.yaml
├── scripts/
│   ├── 02_make_4class_labeling_dataset.py
│   ├── 03_auto_make_4class_labels.py
│   ├── 04_train_4class_yolo.py
│   ├── 05_camera_test.py
│   ├── 06_yolo_webcam_test.py
│   └── 07_dual_model_webcam.py
├── classes_4class.txt
├── LABELING_GUIDE_4CLASS.txt
└── 실행.TXT
```

## Included Files

GitHub에는 코드와 설정 파일만 업로드했습니다.

- `scripts/`: 데이터셋 생성, 자동 라벨링, 학습, 웹캠 테스트 코드
- `dataset/data.yaml`: YOLO 학습용 데이터셋 설정
- `classes_4class.txt`: 클래스 이름 목록
- `LABELING_GUIDE_4CLASS.txt`: 4클래스 라벨링 기준
- `실행.TXT`: 실행 흐름 메모

## Excluded Files

용량이 크거나 로컬 환경에서 생성되는 파일은 저장소에서 제외했습니다.

- `raw_data/`: 원본 데이터
- `dataset_auto_4class/`: 자동 생성 학습 데이터셋
- `labeling_sample/`: 라벨링 샘플 이미지
- `runs/`: YOLO 학습 결과
- `models/`: 학습된 모델 파일
- `*.pt`: YOLO 가중치 파일
- `abvd.ipynb`: 큰 실험용 노트북

필요한 데이터셋과 모델 파일은 로컬에서 별도로 준비해야 합니다.

## Requirements

Python 환경에서 다음 패키지가 필요합니다.

```bash
pip install ultralytics opencv-python
```

프로젝트 환경에 따라 `torch`, `numpy`, `matplotlib` 등이 추가로 필요할 수 있습니다.

## Basic Usage

### 1. 데이터셋 설정 확인

`dataset/data.yaml`의 경로와 클래스 수를 확인합니다.

```yaml
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  - wheelchair
  - wheelchair_user
  - non_wheelchair_person
  - other_object
```

### 2. 모델 학습

```bash
python scripts/04_train_4class_yolo.py
```

### 3. 웹캠 테스트

기본 웹캠 테스트:

```bash
python scripts/06_yolo_webcam_test.py
```

커스텀 모델과 기본 YOLO 모델을 함께 사용하는 테스트:

```bash
python scripts/07_dual_model_webcam.py
```

## Notes

- GitHub에는 큰 데이터와 모델 파일을 올리지 않았습니다.
- 학습이나 테스트를 실행하기 전에 로컬에 데이터셋과 `.pt` 모델 파일이 있는지 확인하세요.
- Windows 경로 또는 한글 폴더명 때문에 문제가 생기면 프로젝트를 영문 경로로 옮겨 실행하는 것을 권장합니다.
