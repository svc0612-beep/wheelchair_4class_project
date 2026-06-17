from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import shutil
import yaml
from collections import Counter
from tqdm import tqdm


# =========================================================
# 0. 경로 설정
# =========================================================

PROJECT_DIR = Path(r"C:\Users\USER\OneDrive\바탕 화면\wheelchair_4class_project")

# 원본 Roboflow 1클래스 데이터셋
RAW_DIR = PROJECT_DIR / "raw_data" / "wheelchair-1"

# 자동 라벨링 결과 저장 폴더
OUT_DIR = PROJECT_DIR / "dataset_auto_4class"

# 기존 wheelchairs 라벨을 사용할지 여부
USE_EXISTING_WHEELCHAIR_LABELS = True

# 전체 데이터 다 처리할지 여부
# None이면 전체 처리
# 테스트만 하고 싶으면 예: 300
MAX_IMAGES_PER_SPLIT = None

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# =========================================================
# 1. 최종 클래스 설계
# =========================================================

CLASS_NAMES = {
    0: "wheelchair",
    1: "wheelchair_user",
    2: "non_wheelchair_person",
    3: "other_object"
}

# YOLOv8n 기본 COCO 클래스 ID
COCO_PERSON = 0

# 휠체어와 헷갈릴 수 있는 주변 사물
# bicycle=1, car=2, motorcycle=3, bus=5, truck=7,
# bench=13, backpack=24, handbag=26, suitcase=28,
# chair=56, couch=57, dining_table=60
OTHER_OBJECT_COCO_IDS = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    13: "bench",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    56: "chair",
    57: "couch",
    60: "dining_table",
}

PERSON_CONF = 0.25
OTHER_CONF = 0.35

# 사람이 기존 wheelchairs 박스와 이 정도 이상 겹치면 wheelchair_user로 판단
INTERSECT_OVER_SMALLER_THRESHOLD = 0.12
IOU_THRESHOLD_FOR_USER = 0.03

# 최종 라벨 중복 제거 기준
DUP_IOU_THRESHOLD = 0.60


# =========================================================
# 2. 기본 함수
# =========================================================

def xywh_norm_to_xyxy_abs(x, y, w, h, img_w, img_h):
    """
    YOLO normalized xywh -> absolute xyxy
    """
    x_center = x * img_w
    y_center = y * img_h
    box_w = w * img_w
    box_h = h * img_h

    x1 = x_center - box_w / 2
    y1 = y_center - box_h / 2
    x2 = x_center + box_w / 2
    y2 = y_center + box_h / 2

    return [x1, y1, x2, y2]


def xyxy_abs_to_xywh_norm(box, img_w, img_h):
    """
    absolute xyxy -> YOLO normalized xywh
    """
    x1, y1, x2, y2 = box

    # 이미지 범위 안으로 보정
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)

    x_center = x1 + box_w / 2
    y_center = y1 + box_h / 2

    return [
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h
    ]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a, box_b):
    inter = intersection_area(box_a, box_b)
    area_a = box_area(box_a)
    area_b = box_area(box_b)

    union = area_a + area_b - inter

    if union <= 0:
        return 0

    return inter / union


def intersect_over_smaller(box_a, box_b):
    inter = intersection_area(box_a, box_b)
    smaller = min(box_area(box_a), box_area(box_b))

    if smaller <= 0:
        return 0

    return inter / smaller


def union_box(box_a, box_b):
    return [
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3])
    ]


def is_valid_box(box, min_size=5):
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def read_existing_wheelchair_boxes(label_path, img_w, img_h):
    """
    기존 Roboflow wheelchairs 라벨을 읽는다.
    bbox 형식이면 그대로 사용.
    polygon 형식이면 polygon을 bbox로 변환.
    """
    boxes = []

    if not label_path.exists():
        return boxes

    text = label_path.read_text(encoding="utf-8").strip()

    if not text:
        return boxes

    for line in text.splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = parts[0]

        # 기존 데이터는 class 0 = wheelchairs
        if class_id != "0":
            continue

        # 정상 YOLO bbox: class x y w h
        if len(parts) == 5:
            try:
                x, y, w, h = map(float, parts[1:])
                box = xywh_norm_to_xyxy_abs(x, y, w, h, img_w, img_h)

                if is_valid_box(box):
                    boxes.append(box)

            except Exception:
                continue

        # polygon 형식: class x1 y1 x2 y2 ...
        elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
            try:
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]

                x_min = min(xs) * img_w
                x_max = max(xs) * img_w
                y_min = min(ys) * img_h
                y_max = max(ys) * img_h

                box = [x_min, y_min, x_max, y_max]

                if is_valid_box(box):
                    boxes.append(box)

            except Exception:
                continue

    return boxes


def remove_duplicate_labels(labels):
    """
    labels 형식:
    [
        {"class_id": 1, "box": [x1,y1,x2,y2], "conf": 0.9}
    ]
    """
    if not labels:
        return []

    # 우선순위:
    # wheelchair_user > wheelchair > non_wheelchair_person > other_object
    class_priority = {
        1: 0,
        0: 1,
        2: 2,
        3: 3
    }

    labels = sorted(
        labels,
        key=lambda d: (
            class_priority.get(d["class_id"], 9),
            -d.get("conf", 1.0),
            -box_area(d["box"])
        )
    )

    kept = []

    for item in labels:
        duplicate = False

        for kept_item in kept:
            if iou(item["box"], kept_item["box"]) >= DUP_IOU_THRESHOLD:
                duplicate = True
                break

        if not duplicate:
            kept.append(item)

    return kept


# =========================================================
# 3. 출력 폴더 준비
# =========================================================
# 중요:
# 기존 dataset_auto_4class를 삭제하지 않는다.
# 중간에 끊겨도 이어서 작업하기 위해 기존 결과를 보존한다.

if not RAW_DIR.exists():
    raise FileNotFoundError(f"원본 데이터 폴더가 없습니다: {RAW_DIR}")

for split in ["train", "valid", "test"]:
    (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

print("출력 폴더 준비 완료:")
print(OUT_DIR)


# =========================================================
# 4. YOLO 기본 모델 로드
# =========================================================

print("\nYOLO 기본 모델 로드 중...")
coco_model = YOLO("yolov8n.pt")
print("YOLO 기본 모델 로드 완료")


# =========================================================
# 5. split별 자동 라벨링
# =========================================================

total_counter = Counter()
split_stats = {}

for split in ["train", "valid", "test"]:
    print(f"\n========== {split} 처리 시작 ==========")

    src_img_dir = RAW_DIR / split / "images"
    src_lbl_dir = RAW_DIR / split / "labels"

    dst_img_dir = OUT_DIR / split / "images"
    dst_lbl_dir = OUT_DIR / split / "labels"

    image_paths = []

    for ext in IMAGE_EXTS:
        image_paths.extend(list(src_img_dir.glob(f"*{ext}")))

    image_paths = sorted(image_paths)

    if MAX_IMAGES_PER_SPLIT is not None:
        image_paths = image_paths[:MAX_IMAGES_PER_SPLIT]

    print(f"{split} 전체 이미지 수:", len(image_paths))

    split_counter = Counter()
    processed = 0
    skipped = 0
    empty_label_images = 0

    for img_path in tqdm(image_paths):
        # =================================================
        # 이어하기 기능:
        # 이미 이미지와 라벨 파일이 둘 다 있으면 건너뜀
        # =================================================
        dst_img_path = dst_img_dir / img_path.name
        dst_label_path = dst_lbl_dir / f"{img_path.stem}.txt"

        if dst_img_path.exists() and dst_label_path.exists():
            skipped += 1

            # 이미 만들어진 라벨 파일도 통계에 포함
            text = dst_label_path.read_text(encoding="utf-8").strip()
            if text == "":
                empty_label_images += 1
            else:
                for line in text.splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        split_counter[int(float(parts[0]))] += 1
                        total_counter[int(float(parts[0]))] += 1
            continue

        # 이미지 크기 확인
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
        except Exception:
            print("이미지 열기 실패:", img_path)
            continue

        # 원본 이미지 복사
        shutil.copy2(img_path, dst_img_path)

        label_path = src_lbl_dir / f"{img_path.stem}.txt"

        # 기존 wheelchairs 박스 읽기
        wheelchair_candidate_boxes = []

        if USE_EXISTING_WHEELCHAIR_LABELS:
            wheelchair_candidate_boxes = read_existing_wheelchair_boxes(
                label_path,
                img_w,
                img_h
            )

        # YOLO 기본 모델로 person / other_object 탐지
        yolo_results = coco_model.predict(
            source=str(img_path),
            conf=0.25,
            verbose=False
        )

        result = yolo_results[0]

        person_boxes = []
        other_boxes = []

        if result.boxes is not None:
            for box in result.boxes:
                coco_cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                if not is_valid_box(xyxy):
                    continue

                if coco_cls == COCO_PERSON and conf >= PERSON_CONF:
                    person_boxes.append({
                        "box": xyxy,
                        "conf": conf
                    })

                elif coco_cls in OTHER_OBJECT_COCO_IDS and conf >= OTHER_CONF:
                    other_boxes.append({
                        "box": xyxy,
                        "conf": conf,
                        "name": OTHER_OBJECT_COCO_IDS[coco_cls]
                    })

        final_labels = []
        used_person_indices = set()

        # =================================================
        # A. 기존 wheelchairs 후보를 wheelchair 또는 wheelchair_user로 분류
        # =================================================

        for wc_box in wheelchair_candidate_boxes:
            matched_person_idx = None
            best_score = 0

            for idx, person in enumerate(person_boxes):
                p_box = person["box"]

                ios = intersect_over_smaller(wc_box, p_box)
                box_iou = iou(wc_box, p_box)

                # 사람이 휠체어 박스와 충분히 겹치면 wheelchair_user
                score = max(ios, box_iou)

                if ios >= INTERSECT_OVER_SMALLER_THRESHOLD or box_iou >= IOU_THRESHOLD_FOR_USER:
                    if score > best_score:
                        best_score = score
                        matched_person_idx = idx

            if matched_person_idx is not None:
                # 휠체어 탄 사람: 사람 + 휠체어 전체를 하나의 박스로 합침
                person_box = person_boxes[matched_person_idx]["box"]
                merged_box = union_box(wc_box, person_box)

                final_labels.append({
                    "class_id": 1,
                    "box": merged_box,
                    "conf": 1.0
                })

                used_person_indices.add(matched_person_idx)

            else:
                # 빈 휠체어 후보
                final_labels.append({
                    "class_id": 0,
                    "box": wc_box,
                    "conf": 1.0
                })

        # =================================================
        # B. 남은 사람은 non_wheelchair_person으로 라벨링
        # =================================================

        for idx, person in enumerate(person_boxes):
            if idx in used_person_indices:
                continue

            p_box = person["box"]

            # 이미 만들어진 wheelchair_user와 너무 겹치면 제외
            overlap_with_existing = False

            for lab in final_labels:
                if lab["class_id"] == 1 and iou(p_box, lab["box"]) > 0.20:
                    overlap_with_existing = True
                    break

            if overlap_with_existing:
                continue

            final_labels.append({
                "class_id": 2,
                "box": p_box,
                "conf": person["conf"]
            })

        # =================================================
        # C. other_object 라벨링
        # =================================================

        for obj in other_boxes:
            obj_box = obj["box"]

            # 휠체어 / 휠체어 사용자 / 사람과 너무 겹치면 제외
            too_overlap = False

            for lab in final_labels:
                if iou(obj_box, lab["box"]) > 0.30:
                    too_overlap = True
                    break

            if too_overlap:
                continue

            final_labels.append({
                "class_id": 3,
                "box": obj_box,
                "conf": obj["conf"]
            })

        # 중복 제거
        final_labels = remove_duplicate_labels(final_labels)

        # =================================================
        # D. YOLO txt 저장
        # =================================================

        label_lines = []

        for lab in final_labels:
            class_id = lab["class_id"]
            x, y, w, h = xyxy_abs_to_xywh_norm(lab["box"], img_w, img_h)

            # 너무 작은 박스 제거
            if w <= 0 or h <= 0:
                continue

            label_lines.append(
                f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )

            split_counter[class_id] += 1
            total_counter[class_id] += 1

        if len(label_lines) == 0:
            empty_label_images += 1

        dst_label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8"
        )

        processed += 1

    split_stats[split] = {
        "processed": processed,
        "skipped": skipped,
        "empty_label_images": empty_label_images,
        "counter": dict(split_counter)
    }

    print(f"\n{split} 처리 완료")
    print("새로 처리한 이미지:", processed)
    print("이미 처리되어 건너뛴 이미지:", skipped)
    print("빈 라벨 이미지:", empty_label_images)
    print("클래스별 라벨 수:", dict(split_counter))


# =========================================================
# 6. data.yaml 생성
# =========================================================

data_yaml = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 4,
    "names": [
        "wheelchair",
        "wheelchair_user",
        "non_wheelchair_person",
        "other_object"
    ]
}

yaml_path = OUT_DIR / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

print("\n========== data.yaml 생성 완료 ==========")
print(yaml_path)
print(yaml_path.read_text(encoding="utf-8"))


# =========================================================
# 7. 결과 요약 저장
# =========================================================

summary_path = OUT_DIR / "auto_label_summary.txt"

lines = []
lines.append("자동 4클래스 라벨링 결과 요약\n")
lines.append(f"원본 데이터: {RAW_DIR}\n")
lines.append(f"출력 데이터: {OUT_DIR}\n\n")

for split, stat in split_stats.items():
    lines.append(f"[{split}]\n")
    lines.append(f"새로 처리한 이미지: {stat['processed']}\n")
    lines.append(f"이미 처리되어 건너뛴 이미지: {stat['skipped']}\n")
    lines.append(f"빈 라벨 이미지: {stat['empty_label_images']}\n")
    lines.append(f"클래스별 라벨 수: {stat['counter']}\n\n")

lines.append("[전체 클래스별 라벨 수]\n")
for class_id in range(4):
    lines.append(f"{class_id} {CLASS_NAMES[class_id]}: {total_counter[class_id]}\n")

summary_path.write_text("".join(lines), encoding="utf-8")

print("\n========== 전체 클래스별 라벨 수 ==========")
for class_id in range(4):
    print(class_id, CLASS_NAMES[class_id], ":", total_counter[class_id])

print("\n요약 파일 저장:")
print(summary_path)

print("\n자동 라벨링 완료")