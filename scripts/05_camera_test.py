import cv2

# =========================================================
# 1. 카메라 번호 설정
# =========================================================
# 0번이 안 되면 1, 2로 바꿔서 테스트
CAMERA_INDEX = 1

# =========================================================
# 2. 웹캠 열기
# =========================================================
# Windows에서 MSMF 오류가 날 수 있어서 DirectShow 방식 사용
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    print("CAMERA_INDEX를 1 또는 2로 바꿔보세요.")
    raise SystemExit

print("웹캠 원본 테스트 시작")
print("q 키를 누르면 종료됩니다.")
print("카메라 번호:", CAMERA_INDEX)

# =========================================================
# 3. 웹캠 화면 출력
# =========================================================
while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽지 못했습니다.")
        break

    print("프레임 평균 밝기:", frame.mean())

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("웹캠 테스트 종료")
