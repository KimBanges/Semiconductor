import cv2
import cvzone
import math
from ultralytics import YOLO

# 이미지 파일 경로
image_path = r"C:\Users\user\Desktop\Semiconductor.v1i.yolov8\test\images\Chip1.webp"

# 이미지 읽기
img = cv2.imread(image_path)

# YOLO 모델 초기화
model = YOLO("Yolo-Weight\\Semiconductor2.pt")

# 반도체 클래스 이름 정의
classNames = ['Capacitor', 'IC chip', 'Transistor', 'Wafer']

# YOLO 모델로 객체 탐지 수행
results = model(img, stream=True)

# 반도체 클래스별 카운트를 저장할 딕셔너리
class_counts = {name: 0 for name in classNames}

# 각 프레임에서 검출된 객체의 클래스를 확인하고 카운트 수행
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # 주황색 바운딩 박스 그리기
        cv2.rectangle(img, (x1, y1), (x2, y2), (24, 215, 24), 2)  # 주황색, 두께 2

        conf = math.ceil((box.conf[0] * 100)) / 100
        class_index = int(box.cls[0])  # 객체의 클래스 인덱스

        if class_index < len(classNames):
            class_name = classNames[class_index]
            if class_name in class_counts:
                class_counts[class_name] += 1

        cvzone.putTextRect(
            img,
            f"{classNames[class_index]}:{conf}",
            (max(0, x1), max(35, y1)),
            scale=1,
            thickness=2,
            colorR=(24, 215, 24),
        )

# 반도체 종류와 개수를 표시할 리스트
count_lines = [f"{class_name}: {count}" for class_name, count in class_counts.items()]

# 화면 오른쪽 상단에 반도체 종류와 개수를 표시
start_y = 40
line_spacing = 30
for i, line in enumerate(count_lines):
    y_offset = start_y + i * line_spacing
    cv2.putText(
        img,
        line,
        (img.shape[1] - 160, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 163, 255),
        2,
    )

# 이미지를 화면에 표시
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
