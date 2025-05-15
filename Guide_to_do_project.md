## 1. Hiểu mục tiêu dự án và chuẩn bị video
- Dự án này nhằm xây dựng hệ thống phân tích bóng rổ sử dụng AI, kết hợp các mô hình như YOLO và thư viện như OpenCV, Supervision.
- Mục tiêu là phân tích video trận đấu bóng rổ để rút ra thông tin chuyên sâu như:
  - Các đường chuyền bóng
  - Các pha cắt bóng
  - Tỷ lệ giữ bóng
  - Tốc độ, khoảng cách di chuyển của cầu thủ
  - Thậm chí chuyển góc quay sang bản đồ chiến thuật nhìn từ trên cao.
- Bạn cần tải về 3 video đầu vào cụ thể từ GitHub (đã được đính kèm trong phần mô tả).
- Các video này tương ứng với 3 tình huống:
  - Chạy với bóng → tốt cho phân tích tốc độ/khoảng cách/giữ bóng
  - Chuyền bóng → phục vụ logic chuyền bóng
  - Cắt bóng → phục vụ logic ngăn chặn
- Tạo một thư mục mới cho dự án, bên trong tạo thư mục con tên input_videos → để chứa 3 video trên.
- Mở thư mục dự án bằng code editor (như Visual Studio Code).
## 2. Cài đặt mô hình nhận diện ban đầu với YOLOv8
- Bước cốt lõi đầu tiên là dùng YOLO để phát hiện cầu thủ và bóng trong từng khung hình của video.
- YOLO là viết tắt của "You Only Look Once" — một mô hình nhận diện đối tượng mạnh mẽ, hiện đã phát triển đến phiên bản V11.
- Thư viện Ultralytics hỗ trợ chạy các phiên bản YOLO từ V1 đến V11 rất tiện lợi.
Cài đặt Ultralytics bằng lệnh:
```bash
pip install ultralytics
```
- Tạo file main.py
- Import YOLO từ Ultralytics:
```bash
from ultralytics import YOLO
```
- Tải mô hình YOLOv8X đã huấn luyện sẵn:
```bash
model = YOLO("yolov8x.pt") (or yolov8x). 
```
  - Các phiên bản YOLOv8 gồm: nano, small, medium, large, X-large
  - X-large (yolov8x) cho độ chính xác cao nhất nhưng đòi hỏi tài nguyên mạnh hơn (RAM, GPU)
- Chạy mô hình trên video đầu vào:
```bash
model.predict(source="input_videos/video_1.mp4", save=True)
```
  - Lệnh trên sẽ xử lý từng khung hình và lưu lại video kết quả.
  - predict() trả về kết quả nhận diện: tọa độ bounding box, độ tin cậy (confidence), nhãn lớp (như "person", "sports ball").
  - In kết quả khung hình đầu tiên:
- In kết quả khung hình đầu tiên:
```bash
results = model.predict(...)
print(results[0].boxes)
```
- Video đầu ra sẽ nằm trong thư mục: runs/detect/predict.
-  Kiểm tra video kết quả: thấy mô hình nhận diện được người (kể cả khán giả - không mong muốn) và bóng còn chưa ổn định.
## 3. Fine-tuning YOLO for Specific Object Detection (Players and Balls)
### Mục tiêu:
Để tăng độ chính xác và đảm bảo mô hình chỉ nhận diện đúng các đối tượng như cầu thủ trong sân và bóng rổ, bạn cần fine-tune (huấn luyện lại) mô hình YOLO trên một tập dữ liệu được gán nhãn tùy chỉnh.
### Hiểu rõ về Fine-tuning:
- Fine-tuning là một dạng transfer learning – bạn tận dụng mô hình đã được huấn luyện trước, rồi "huấn luyện lại nhẹ" để nó phù hợp hơn với bài toán cụ thể của mình.
- Cần một tập dữ liệu có ảnh và bounding box xác định vị trí các đối tượng bạn muốn nhận diện, ví dụ: `player`, `referee`, `ball`, `hoop`
- Roboflow là nền tảng được đề xuất để tìm kiếm hoặc tạo các tập dữ liệu có gán nhãn.
### YOLOv5 phù hợp hơn YOLOv8 cho sports analytics:
- Mặc dù `YOLOv8` hiện đại hơn, `YOLOv5` cho kết quả ổn định hơn trong việc phát hiện cầu thủ và bóng sau fine-tuning.
- Huấn luyện mô hình sâu yêu cầu GPU → Dùng Google Colab là giải pháp tối ưu (miễn phí, có GPU như T4).
### Các bước huấn luyện trên Google Colab:
#### 1. Cài thư viện: 
```bash
!pip install roboflow ultralytics
```
#### 2. Tải dataset từ Roboflow
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("basketball-players")
dataset = project.version(17).download("yolov5")
```
#### 3. Điều chỉnh cấu trúc thư mục (nếu cần):
```bash
import shutil
shutil.move('basketball-players-17', 'data')
```
#### 4. Huấn luyện YOLOv5:
```bash
!yolo task=detect mode=train model=yolov5l6u.pt data=data/data.yaml epochs=100 imgsz=640 batch=8
# yolov5l6u.pt là mô hình lớn, tối ưu cho độ chính xác.
```
#### 5. Tải file trọng số tốt nhất:
- Tìm file:`runs/detect/train/weights/best.pt`
- Tải về → Đặt vào thư mục `models/` trong dự án
- Đổi tên theo mục đích:
  - `player_detector.pt`
  - `ball_detector_model.pt`
- Lặp lại các bước để huấn luyện mô hình riêng cho quả bóng:
  - Sử dụng cùng tập dữ liệu hoặc một tập riêng về bóng.
  - Có thể tăng `epochs=250` để cải thiện hiệu suất nhận diện bóng.
### Sử dụng mô hình fine-tuned trong code:
- Trong main.py, thay đoạn tải model YOLO như sau:
```bash
player_model = YOLO("models/player_detector.pt")
ball_model = YOLO("models/ball_detector_model.pt")
```
- Khi đã có 2 mô hình riêng biệt, bạn có thể áp dụng các kỹ thuật như:
  - lọc vị trí cầu thủ trong sân
  - tách riêng bóng khỏi khán giả
  - xây bản đồ chiến thuật (top-down tactical map) chính xác hơn
## 4. Set up Core Code Structure and Video Handling
### Mục tiêu:
- Tổ chức code gọn gàng, dễ mở rộng.
- Viết các hàm tiện ích để đọc và lưu video bằng OpenCV.
### Cấu trúc thư mục:
```bash
project_folder/
│
├── main.py
├── models/
├── input_videos/
├── utils/
│   ├── __init__.py
│   └── video_utils.py
```
### 1. Dọn dẹp main.py
- Xóa đoạn code demo ban đầu (nếu có).
- Chuyển sang sử dụng main() chuẩn hóa.
### 2. Tạo thư mục tiện ích utils/
```bash
mkdir utils
touch utils/__init__.py
touch utils/video_utils.py
```
### 3. Cài thư viện xử lý video:
```bash
pip install opencv-python
```
### 4. Viết hàm đọc và lưu video trong `video_utils.py`
```bash
import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_frames, output_path, fps=30):
    if not output_frames:
        raise ValueError("No frames to save")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    height, width, _ = output_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in output_frames:
        out.write(frame)

    out.release()
```
### 5. Kết nối tiện ích trong `utils/__init__.py`
```bash
from .video_utils import read_video, save_video
```
### 6. Viết lại `main.py` để test
```bash
from utils import read_video, save_video

def main():
    input_path = "input_videos/video_1.mp4"
    output_path = "output_videos/test_output.avi"

    frames = read_video(input_path)
    print(f"Số lượng frame: {len(frames)}")

    save_video(frames, output_path)
    print("Video đã được lưu.")

if __name__ == "__main__":
    main()
```
### Kết quả mong đợi:
- Video video_1.mp4 được đọc và lưu lại thành test_output.avi trong thư mục `output_videos/.`
- Điều này kiểm tra rằng `read_video` và `save_video` hoạt động ổn.

## 5. Tracking đối tượng qua các khung hình (Player Tracking)
### Mục tiêu:
- Sau khi YOLO detect được player, ta cần track họ xuyên suốt video.
- Gán ID cố định cho từng người chơi để tính tốc độ, khoảng cách, v.v.
### 1. Tạo module trackers/
```bash
mkdir trackers
touch trackers/__init__.py
touch trackers/player_tracker.py
```
### 2. Cài thư viện Supervision:
```bash
pip install supervision
```
### 3. Viết class `PlayerTracker` trong `player_tracker.py`
```bash
from ultralytics import YOLO
import supervision as sv

class PlayerTracker:
    def __init__(self, model_path="models/player_detector.pt"):
        # Load YOLO model đã fine-tuned để detect player
        self.model = YOLO(model_path)
        # Khởi tạo tracker (ByteTrack)
        self.tracker = sv.ByteTrack()

    def detect_and_track_frames(self, frames, batch_size=20):
        tracked_frames = []
        # Chia frame thành các batch để xử lý hiệu quả hơn
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            # Dự đoán bằng YOLO
            results = self.model.predict(batch, imgsz=640, verbose=False)
            for result in results:
                detections = sv.Detections.from_ultralytics(result)
                tracked = self.tracker.update_with_detections(detections)
                # Lưu thông tin: bounding box + ID track
                frame_data = []
                for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                    x1, y1, x2, y2 = box
                    frame_data.append({
                        "track_id": int(track_id),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
                tracked_frames.append(frame_data)
        return tracked_frames
```
### 4. Kết quả đầu ra:
- Hàm detect_and_track_frames() sẽ trả về một list chứa thông tin tracking cho từng frame:
```bash
[
  [  # Frame 1
    {"track_id": 1, "bbox": [100, 200, 180, 300]},
    {"track_id": 2, "bbox": [220, 210, 280, 320]},
  ],
  [  # Frame 2
    {"track_id": 1, "bbox": [102, 203, 182, 302]},
    {"track_id": 2, "bbox": [223, 213, 283, 323]},
  ],
  ...
]
```
## 6. Thêm Stub Logic để checkpoint kết quả trung gian
### Mục tiêu:
- Video dài => mỗi lần chạy lại detect/track rất tốn thời gian.
- Ta sẽ lưu kết quả ra file .pkl (pickle) sau khi xử lý.
- Lần chạy sau chỉ cần load lại, không cần detect/track lại nữa.
###  1. Tạo `stubs_utils.py` trong `utils`
```bash
touch utils/stubs_utils.py
```
### 2. Cài đặt pickle (nếu cần):
```bash
pip install pickle5
```
### 3. Code trong `utils/stubs_utils.py`
```bash
import os
import pickle

def save_stub(stub_path, obj):
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(obj, f)

def read_stub(stub_path):
    if not os.path.exists(stub_path):
        return None
    with open(stub_path, "rb") as f:
        return pickle.load(f)
```
### 4. Trong utils/__init__.py, thêm:
```bash
from .stubs_utils import save_stub, read_stub
```
### 5. Tích hợp stub vào PlayerTracker
Giả sử bạn có phương thức `detect_and_track_frames` như bước trước, sửa lại như sau:
```bash
def detect_and_track_frames(self, frames, stub_path=None, read_from_stub=True):
    # Nếu có stub và cho phép đọc
    if read_from_stub and stub_path:
        cached = read_stub(stub_path)
        if cached and len(cached) == len(frames):
            print(f"✔ Loaded tracking data from {stub_path}")
            return cached
    # Nếu không có hoặc không hợp lệ → xử lý lại
    print("⚙ Detecting & tracking players...")
    tracked_frames = []
    for i in range(0, len(frames), 20):
        batch = frames[i:i+20]
        results = self.model.predict(batch, imgsz=640, verbose=False)
        for result in results:
            detections = sv.Detections.from_ultralytics(result)
            tracked = self.tracker.update_with_detections(detections)

            frame_data = []
            for box, track_id in zip(tracked.xyxy, tracked.tracker_id):
                x1, y1, x2, y2 = box
                frame_data.append({
                    "track_id": int(track_id),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
            tracked_frames.append(frame_data)
    # Lưu lại kết quả sau xử lý
    if stub_path:
        save_stub(stub_path, tracked_frames)
        print(f"💾 Saved tracking data to {stub_path}")
    return tracked_frames
```

### 6. Cách sử dụng trong main.py
```bash
from trackers.player_tracker import PlayerTracker
from utils import read_video

def main():
    frames = read_video("input_videos/video_1.mp4")
    tracker = PlayerTracker()

    tracked_frames = tracker.detect_and_track_frames(
        frames,
        stub_path="stubs/video1_tracking.pkl",
        read_from_stub=True
    )
    # Tiếp tục xử lý...
```
###  Lợi ích:
- Chạy lần đầu: detect & track → lưu stub.
- Chạy lần sau: load lại kết quả → nhanh gấp 10x+.
## 7: Custom Annotation – Vẽ bounding box
### Mục tiêu:
- Thay vì các bounding box mặc định, ta sẽ vẽ ellipse bên dưới người chơi + ID.
- Dễ xem hơn, giống phong cách các hệ thống phân tích thể thao thực tế.
- Create a drawers folder and an __init__.py file.
### 1. Tạo cấu trúc thư mục:
```bash
mkdir drawers
touch drawers/__init__.py
touch drawers/player_tracks_drawer.py

mkdir utils
touch utils/bbox_utils.py  # nếu chưa có
```
### 2. PlayerTracksDrawer – Vẽ dữ liệu track
```bash
from utils.bbox_utils import draw_ellipse

class PlayerTracksDrawer:
    def __init__(self):
        pass

    def draw(self, frames, player_tracks):
        annotated_frames = []

        for frame, tracks in zip(frames, player_tracks):
            frame_copy = frame.copy()

            for track in tracks:
                bbox = track["bbox"]
                track_id = track["track_id"]
                frame_copy = draw_ellipse(frame_copy, bbox, track_id)

            annotated_frames.append(frame_copy)

        return annotated_frames
```
### 3. draw_ellipse() – Vẽ ellipse + ID
```bash
import cv2

def draw_ellipse(frame, bbox, track_id=None, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    center = (int((x1 + x2) / 2), int(y2))  # đáy của bounding box
    axes = (int((x2 - x1) / 2), 10)  # chiều ngang/vertical ellipse

    cv2.ellipse(frame, center, axes, 0, 0, 360, color, -1)

    if track_id is not None:
        text = f"ID: {track_id}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame
```
### 4. Thêm import vào __init__.py:
- `drawers/__init__.py`: 
```bash
from .player_tracks_drawer import PlayerTracksDrawer
```
- `utils/__init__.py` (nếu cần):
```bash
from .bbox_utils import draw_ellipse
```
### 5. Gọi vẽ trong main.py:
- Sau khi track xong:
```bash
from drawers import PlayerTracksDrawer

# Sau khi detect & track:
drawer = PlayerTracksDrawer()
annotated_frames = drawer.draw(frames, tracked_frames)

# Lưu video mới
save_video(annotated_frames, "outputs/video_1_annotated.avi")
```
### Kết quả:
- Mỗi cầu thủ sẽ có một vòng tròn (ellipse) phía dưới người để chỉ vị trí.
- Có cả Track ID để phân biệt từng người.
- Tăng tính chuyên nghiệp và dễ quan sát cho video phân tích trận đấu.
## 8. Tinh chỉnh logic phát hiện bóng
### Mục tiêu:
- Trong mỗi frame, có thể có nhiều object được phát hiện là bóng (hoặc sai lệch).
- Cần lọc ra một bóng duy nhất có độ tin cậy cao nhất.
### 1. Tạo file ball_tracker.py
```bash
mkdir trackers  # nếu chưa tạo
touch trackers/ball_tracker.py
```
### 2. Logic chọn bóng tốt nhất cho mỗi frame:
```bash
from ultralytics import YOLO
import supervision as sv

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_and_track_ball(self, frames):
        all_ball_detections = []

        results = self.model.predict(frames, stream=True, verbose=False)

        for frame_result in results:
            balls = []
            for box in frame_result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.model.names[cls]

                if label == 'sports ball':  # hoặc 'ball' tùy model
                    balls.append({
                        'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        'conf': conf
                    })

            # Chọn quả bóng có độ tin cậy cao nhất
            if balls:
                best_ball = max(balls, key=lambda x: x['conf'])
                all_ball_detections.append(best_ball)
            else:
                all_ball_detections.append(None)

        return all_ball_detections
```
### 3. Cách sử dụng trong main.py
```bash
from trackers.ball_tracker import BallTracker

ball_tracker = BallTracker("models/ball_detector_model.pt")
ball_detections = ball_tracker.detect_and_track_ball(frames)
```
### 4. Kết quả:
- Mỗi frame sẽ trả về:
- None nếu không có bóng.
- Hoặc {'bbox': [...], 'conf': ...} cho quả bóng có độ tin cậy cao nhất.
### Gợi ý mở rộng:
- Vẽ bóng bằng cv2.circle() thay vì bbox để tạo hiệu ứng đẹp.
- Dùng logic track_id nếu muốn theo dõi quả bóng di chuyển theo thời gian.
## 9. Gán cầu thủ vào đội bằng phân loại không huấn luyện (Zero-Shot Classification)
### Mục tiêu:
- Phân loại cầu thủ vào 1 trong 2 đội dựa vào màu áo (ví dụ: "áo trắng", "áo xanh đậm") mà không cần huấn luyện mô hình mới.
### Cài đặt cần thiết
```bash
pip install transformers torchvision pillow
```
### Cấu trúc thư mục
```bash
team_assigner/
├── __init__.py
└── team_assigner.py
```
### Logic gán đội: team_assigner/team_assigner.py
```bash
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class TeamAssigner:
    def __init__(self, device='cpu'):
        self.device = device
        self.classes = ["white shirt", "dark blue shirt"]  # tên đội dựa vào màu áo
        self.model = CLIPModel.from_pretrained("fashionclip/fashion-clip").to(device)
        self.processor = CLIPProcessor.from_pretrained("fashionclip/fashion-clip")

    def get_player_team(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]  # crop người chơi từ frame

        # Convert to PIL Image
        image = Image.fromarray(cropped)

        inputs = self.processor(
            text=self.classes,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # scores
        probs = logits_per_image.softmax(dim=1)       # chuyển thành xác suất

        best_class = self.classes[probs.argmax().item()]
        return best_class  # trả về tên đội
```
### Cách sử dụng trong main.py
```bash
from team_assigner.team_assigner import TeamAssigner

team_assigner = TeamAssigner(device='cuda' if torch.cuda.is_available() else 'cpu')

for frame_idx, frame in enumerate(frames):
    for player in player_tracks[frame_idx]:
        bbox = player['bbox']
        team = team_assigner.get_player_team(frame, bbox)
        player['team'] = team
```
### Kết quả:
- Mỗi cầu thủ sẽ được gán thuộc về “white shirt” hoặc “dark blue shirt”.
- Có thể dùng trường player['team'] để vẽ màu viền khác nhau hoặc thống kê theo đội.
### Mở rộng (nếu cần):
- Gán team_id (0, 1) thay vì string.
- Lưu kết quả này bằng stub (pickle) để không phải phân loại lại mỗi lần chạy.

## 10. Xác định người giữ bóng (Ball Possession Detection)
### Mục tiêu:
Trong hệ thống phân tích trận đấu bóng rổ, một tính năng quan trọng là biết ai đang giữ bóng tại từng thời điểm. Điều này là nền tảng cho các phân tích nâng cao như:
- Tính % kiểm soát bóng của mỗi đội.
- Phát hiện các đường chuyền, steal, rebound.
- Tạo video highlight tự động.
Do đó, bước này sẽ so sánh vị trí bóng với các cầu thủ và xác định ai đang giữ bóng ở mỗi frame.
### Logic phân tích "ball possession"
#### 1. Lấy vị trí trung tâm bóng trong mỗi frame
- Từ kết quả của YOLO ball detector (ball_positions), bạn có toạ độ tâm của bóng trong mỗi frame.
#### 2. Lấy các bounding box của cầu thủ từ player tracker
- Mỗi cầu thủ đã được gán track ID cố định. Mỗi frame có danh sách cầu thủ đang xuất hiện.
#### 3.Tính khoảng cách giữa bóng và các "điểm trọng yếu" của mỗi cầu thủ
- Thay vì chỉ tính từ tâm cầu thủ → bóng, bạn sẽ lấy nhiều điểm đại diện như: bottom center, center, left center, right center từ bounding box.
- Tính khoảng cách từ tâm bóng đến các điểm này.
#### 4. Người giữ bóng = cầu thủ gần bóng nhất
- Nếu khoảng cách < ngưỡng hợp lý (ví dụ dưới 50 pixels), ta xem cầu thủ đó đang giữ bóng.
- Tránh nhấp nháy (flickering) → dùng smoothing
#### 5. Chỉ xem là "đang giữ bóng" nếu cầu thủ đó gần bóng liên tục trong ≥11 frame.
- Tránh tình trạng bóng bị nhầm sang cầu thủ khác chỉ vì một frame lệch.
- Lưu kết quả vào stub để không phải xử lý lại khi chạy lại chương trình.
### Cấu trúc thư mục
```bash
ball_acquisition/
├── __init__.py
└── ball_acquisition_detector.py

utils/
├── geometry_utils.py   # chứa các hàm đo khoảng cách
```
### Cài thư viện cần thiết
```bash
pip install numpy
```
### `utils/geometry_utils.py`
```bash
import numpy as np

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_key_points_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [
        (cx, y2),         # bottom center
        (x1, cy),         # left center
        (x2, cy),         # right center
        (cx, cy)          # center
    ]
```
- Vì sao chọn nhiều điểm đại diện?
    - Trong thực tế, bóng không luôn nằm ở giữa người chơi.
    - Điểm như “bottom center” sẽ gần chân – nơi bóng thường hiện diện khi rê bóng.
### `ball_acquisition/ball_acquisition_detector.py`
```bash
from utils.geometry_utils import euclidean_distance, get_key_points_from_bbox
from utils import save_stub, read_stub

class BallAcquisitionDetector:
    def __init__(self, min_hold_frames=11):
        self.min_hold_frames = min_hold_frames

    def detect_ball_possession(self, frames, player_tracks, ball_positions, read_from_stub=True, stub_path="stubs/ball_possession.pkl"):
        if read_from_stub:
            try:
                result = read_stub(stub_path)
                if result and len(result) == len(frames):
                    return result
            except:
                pass

        possession_history = []
        last_possessor = None
        consecutive_count = 0

        for frame_idx in range(len(frames)):
            ball = ball_positions[frame_idx]
            players = player_tracks[frame_idx]

            if not ball or not players:
                possession_history.append(None)
                continue

            bx, by = ball['center']
            min_distance = float("inf")
            possessor_id = None

            for player in players:
                pid = player['id']
                bbox = player['bbox']
                points = get_key_points_from_bbox(bbox)
                for pt in points:
                    dist = euclidean_distance(pt, (bx, by))
                    if dist < min_distance:
                        min_distance = dist
                        possessor_id = pid

            # smoothing possession
            if possessor_id == last_possessor:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_possessor = possessor_id

            if consecutive_count >= self.min_hold_frames:
                possession_history.append(possessor_id)
            else:
                possession_history.append(None)

        save_stub(stub_path, possession_history)
        return possession_history
```
### Tích hợp trong `main.py`
```bash
from ball_acquisition.ball_acquisition_detector import BallAcquisitionDetector

ball_acquisition = BallAcquisitionDetector()
ball_possessions = ball_acquisition.detect_ball_possession(
    frames,
    player_tracks=player_tracks,
    ball_positions=ball_detections
)
```
###  Output:
- ball_possessions: list ID cầu thủ giữ bóng theo từng frame.
- Dùng để vẽ hiệu ứng, tạo thống kê hoặc phát hiện các sự kiện như pass, steal, shot.

## 11. Tính toán quyền kiểm soát bóng theo đội (Team Ball Control)
Từ kết quả bước 10 (ai giữ bóng ở mỗi frame) và thông tin đội (team assignment) của mỗi cầu thủ, ta sẽ tính xem:
- Đội nào đang kiểm soát bóng ở mỗi frame.
- Tạo numpy array chứa team ID theo từng frame: `1`, `2`, hoặc `-1` (không ai giữ bóng).
- Từ đó tính toán:
    - Tổng thời gian mỗi đội giữ bóng.
    - % kiểm soát bóng toàn trận hoặc theo từng đoạn video.
### Logic xử lý
#### Input:
- `ball_possessions`: list các `player_id` đang giữ bóng tại mỗi frame. (từ bước 10)
- `player_team_map`: dict dạng `{player_id: team_id}` do bước phân loại màu áo sinh ra (`team_assigner`).

#### Output:
- team_ball_control: numpy array chứa team ID kiểm soát bóng tương ứng với mỗi frame:
    - `1` → Team 1
    - `2` → Team 2
    - `-1` → Không rõ (không ai giữ bóng)
#### Cập nhật class BallAcquisitionDetector
File: `ball_acquisition/ball_acquisition_detector.py`
- Hàm mới: `get_team_ball_control`
```bash
import numpy as np

class BallAcquisitionDetector:
    # đã có __init__ và detect_ball_possession từ bước trước

    def get_team_ball_control(self, ball_possessions, player_team_map, neutral_value=-1):
        """
        ball_possessions: list[player_id or None] (theo từng frame)
        player_team_map: dict {player_id: team_id}
        return: numpy array [team_id] theo từng frame
        """
        team_control = []

        for pid in ball_possessions:
            if pid is None or pid not in player_team_map:
                team_control.append(neutral_value)
            else:
                team_control.append(player_team_map[pid])

        return np.array(team_control)
```
### Tích hợp vào main.py
Sau khi đã có:
- ball_possessions từ bước 10.
- player_team_map sau bước team classification (ví dụ: {3: 1, 7: 1, 12: 2, 15: 2}).
```bash
# Ví dụ team assignment: player_id → team_id
player_team_map = {p['id']: p['team'] for frame in player_tracks for p in frame}

# Chạy bước 11:
team_control_array = ball_acquisition.get_team_ball_control(
    ball_possessions,
    player_team_map
)

# In thống kê % kiểm soát bóng
total_frames = len(team_control_array)
team1_time = np.sum(team_control_array == 1)
team2_time = np.sum(team_control_array == 2)

print(f"Team 1 possession: {team1_time / total_frames * 100:.1f}%")
print(f"Team 2 possession: {team2_time / total_frames * 100:.1f}%")
```

## 12.Vẽ Overlay Thống Kê Lên Video
### Mục tiêu của bước này
Hiển thị thông tin kiểm soát bóng của đội bóng (Team Ball Control) trong từng khung hình (frame), trực tiếp trên video:
- Frame nào đang được kiểm soát bởi đội nào? (VD: "Team 1 has the ball")
- Nếu không ai giữ bóng thì hiện "Neutral".
### Logic xử lý
#### Input:
- `frame`: 1 frame trong video.
- `frame_number`: chỉ số của frame hiện tại.
- `team_control_array`: numpy array chứa team ID kiểm soát bóng ở mỗi frame (từ bước 11).
### Output:
 Frame đã được vẽ thêm thông tin overlay:
- Hộp chữ ở góc trên (hoặc dưới)
- Màu sắc đại diện từng đội
### Cập nhật class BallAcquisitionDetector:
Trong `file ball_acquisition/ball_acquisition_detector.py`, thêm:
```bash
import cv2

class BallAcquisitionDetector:
    # ... các hàm trước
    def draw_frame(self, frame, frame_number, team_control_array):
        """
        Vẽ overlay kiểm soát bóng lên frame
        """
        team_id = team_control_array[frame_number]
        if team_id == 1:
            color = (255, 0, 0)  # Đội 1: xanh dương
            text = "Team 1 has the ball"
        elif team_id == 2:
            color = (0, 0, 255)  # Đội 2: đỏ
            text = "Team 2 has the ball"
        else:
            color = (128, 128, 128)  # Không ai: xám
            text = "No team in control"
        # Vẽ background rectangle
        cv2.rectangle(frame, (10, 10), (310, 50), color, -1)
        # Vẽ text overlay
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return frame
```
### Tích hợp vào vòng lặp chính `main.py`:
```bash
ball_acquisition = BallAcquisitionDetector()

output_video_frames = []

for i, frame in enumerate(annotated_frames):
    # vẽ overlay team control
    frame = ball_acquisition.draw_frame(frame, i, team_control_array)
    output_video_frames.append(frame)
```
## 13. Xác định điểm mốc sân đấu
### Mục tiêu
Từ video thật, xác định các keypoints sân bóng (ví dụ: trung tâm sân, 2 vạch 3 điểm, bảng rổ...) – dùng để:
- Làm homography (biến đổi phối cảnh từ camera về góc nhìn từ trên xuống).
- Áp vị trí cầu thủ & bóng vào sơ đồ chiến thuật dễ phân tích.
### Tư duy xử lý:
| Giai đoạn                                | Mô tả                                      |
| ---------------------------------------- | ------------------------------------------ |
| 1. Fine-tune YOLOv8 pose model           | Huấn luyện để phát hiện keypoints sân bóng |
| 2. Dự đoán keypoints mỗi frame           | Trích xuất vị trí các điểm landmark        |
| 3. Tối ưu hóa xử lý qua batch + lưu stub | Tránh lặp lại inference nặng               |

### 1. Fine-tune YOLOv8 Pose Model (trên Colab)
- Dùng model: yolov8x-pose.pt
- Dataset: từ Roboflow (gắn nhãn các điểm sân)
- Command mẫu:
```bash
yolo task=pose mode=train model=yolov8x-pose.pt data=data.yaml epochs=500 imgsz=640 batch=16
```
### 2. Cấu trúc thư mục
```bash
court_keypoint_detector/
├── __init__.py
└── court_keypoint_detector.py
```
### 3. Code: CourtKeypointDetector
```bash
# court_keypoint_detector/court_keypoint_detector.py

from ultralytics import YOLO
import numpy as np
import os
from utils import read_stub, save_stub

class CourtKeypointDetector:
    def __init__(self, model_path="models/best.pt"):
        self.model = YOLO(model_path)

    def get_court_keypoints(self, frames, read_from_stub=True, stub_path="stubs/court_keypoints.pkl"):
        if read_from_stub and os.path.exists(stub_path):
            print("[Stub] Loading court keypoints from:", stub_path)
            return read_stub(stub_path)

        all_keypoints = []

        print("[INFO] Running court keypoint detection...")
        for i in range(0, len(frames), 20):  # xử lý theo batch
            batch = frames[i:i+20]
            results = self.model.predict(batch, stream=True)

            for result in results:
                keypoints = result.keypoints.xy.cpu().numpy()  # shape: [num_kpts, 2]
                all_keypoints.append(keypoints)

        save_stub(stub_path, all_keypoints)
        return all_keypoints
```
### 4. Tích hợp vào main.py
```bash
from court_keypoint_detector import CourtKeypointDetector

# Sau khi tracking cầu thủ xong:
court_detector = CourtKeypointDetector("models/best.pt")
court_keypoints = court_detector.get_court_keypoints(video_frames)

# Debug kiểm tra:
print("[DEBUG] Frame 0 Keypoints:", court_keypoints[0])
```
### Output: Dữ liệu keypoints theo từng frame
```bash
court_keypoints[0]  # -> array([[100.5, 222.1], [345.2, 120.4], ..., [x, y]])
```
- Mỗi frame có 5–10 điểm (tùy số lượng keypoints bạn dán nhãn khi fine-tune).
- Dữ liệu này là đầu vào cực kỳ quan trọng để:
  - Chuyển góc nhìn sân bóng về từ trên xuống.
  - Gắn tọa độ cầu thủ, bóng vào tactical board.

## 14. Hiển thị trực quan các keypoint sân bóng
### Mục tiêu:
- Vẽ các court keypoints (tọa độ landmark sân bóng) lên từng frame của video, bao gồm:
- Dấu chấm tại vị trí keypoint.
### Chiến lược:
- Dùng thư viện Supervision (sv), vốn tương thích tốt với Ultralytics YOLO output.
- Hiển thị keypoints với sv.vertex_annotator và sv.vertex_label_annotator.
- Kết quả là các khung hình đã annotated keypoints, cực hữu ích để:
  - Debug chất lượng dữ liệu.
  - Chuẩn bị mapping sang tactical board.
Số thứ tự keypoint (label) để dễ debug và mapping
### 1. Tạo file `drawers/court_key_points_drawer.py`
```bash
# drawers/court_key_points_drawer.py

import numpy as np
import supervision as sv

class CourtKeyPointsDrawer:
    def __init__(self, keypoint_color=sv.Color.red()):
        self.vertex_annotator = sv.VertexAnnotator(color=keypoint_color, thickness=4)
        self.vertex_label_annotator = sv.VertexLabelAnnotator()

    def draw(self, frames, court_key_points):
        output_frames = []
        for i, frame in enumerate(frames):
            keypoints = court_key_points[i]

            if keypoints is None or len(keypoints) == 0:
                output_frames.append(frame.copy())
                continue
            # Convert to (N, 2) numpy array of int
            keypoints_arr = np.array(keypoints).astype(int)
            # Generate labels: ["0", "1", "2", ...]
            labels = [str(i) for i in range(len(keypoints_arr))]

            # Draw keypoints + labels using Supervision
            frame_copy = frame.copy()
            frame_copy = self.vertex_annotator.annotate(scene=frame_copy, vertices=keypoints_arr)
            frame_copy = self.vertex_label_annotator.annotate(scene=frame_copy, labels=labels, vertices=keypoints_arr)
            output_frames.append(frame_copy)
        return output_frames
```
### 2. Trong `drawers/__init__.py`
```bash
from .court_key_points_drawer import CourtKeyPointsDrawer
```
### 3. Tích hợp vào main.py
Sau khi bạn đã lấy `court_keypoints` từ `CourtKeypointDetector`, thêm:
```bash
from drawers import CourtKeyPointsDrawer
# Initialize Drawer
court_drawer = CourtKeyPointsDrawer()
# Annotate keypoints on video frames
video_frames_with_keypoints = court_drawer.draw(video_frames, court_keypoints)
# Update output frames
output_video_frames = video_frames_with_keypoints
```
### Output kết quả:
- Trên mỗi khung hình, các keypoints sẽ được vẽ như chấm đỏ với nhãn đánh số.
- Hữu ích để:
  - So sánh các keypoints giữa các frame → kiểm tra độ ổn định.
  - Tự động hóa matching với sơ đồ sân mẫu trong bước homography sắp tới.
## 15: Tạo Tactical View bằng Perspective Transformation
### Mục tiêu:
- Chuyển đổi vị trí pixel trong camera view thành tọa độ thực tế/met hoặc tactical pixel map.
- Dùng homography matrix từ OpenCV để chuyển đổi tọa độ từ keypoint trên sân sang bản đồ chiến thuật (ảnh sơ đồ sân).

### 1. Cấu trúc thư mục
```bash
tactical_view_converter/
├── __init__.py
└── tactical_view_converter.py
```
### 2. `tactical_view_converter.py` – Chuyển đổi toạ độ
```bash
# tactical_view_converter.py

import cv2
import numpy as np

class TacticalViewConverter:
    def __init__(self, tactical_court_image_path, court_width_m, court_height_m, tactical_width_px, tactical_height_px):
        self.tactical_court_image_path = tactical_court_image_path
        self.court_width_m = court_width_m
        self.court_height_m = court_height_m
        self.tactical_width_px = tactical_width_px
        self.tactical_height_px = tactical_height_px
        self.homography_matrix = None

    def generate_perspective_transform_matrix(self, camera_keypoints, tactical_keypoints):
        # Convert to np.float32 arrays
        cam_pts = np.array(camera_keypoints, dtype=np.float32)
        tac_pts = np.array(tactical_keypoints, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(cam_pts, tac_pts)

    def convert_pixel_to_tactical(self, points):
        # points: (N, 2)
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.homography_matrix)
        return transformed.reshape(-1, 2)

    def get_player_positions(self, all_player_boxes, court_keypoints):
        # all_player_boxes: List[List[ [x1, y1, x2, y2] ]]
        all_positions = []

        for frame_boxes in all_player_boxes:
            frame_positions = []
            for box in frame_boxes:
                # Use bottom-center point of bbox
                x_center = (box[0] + box[2]) / 2
                y_bottom = box[3]
                tactical_pos = self.convert_pixel_to_tactical([(x_center, y_bottom)])
                frame_positions.append(tactical_pos[0])
            all_positions.append(frame_positions)

        return all_positions
```
### 3. `tactical_view_drawer.py` – Vẽ tactical overlay
```bash
# drawers/tactical_view_drawer.py

import cv2
import numpy as np

class TacticalViewDrawer:
    def __init__(self, overlay_opacity=0.6, point_color=(0, 0, 255)):
        self.overlay_opacity = overlay_opacity
        self.point_color = point_color

    def draw(self, frames, court_img_path, width_px, height_px, tactical_keypoints):
        court_img = cv2.imread(court_img_path)
        court_img = cv2.resize(court_img, (width_px, height_px))

        output_frames = []
        for frame in frames:
            overlay = frame.copy()
            overlay[0:height_px, 0:width_px] = court_img
            # Draw keypoints on court image
            for i, pt in enumerate(tactical_keypoints):
                pt_int = tuple(int(x) for x in pt)
                cv2.circle(overlay, pt_int, 5, self.point_color, -1)
                cv2.putText(overlay, str(i), pt_int, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.point_color, 1)

            blended = cv2.addWeighted(overlay, self.overlay_opacity, frame, 1 - self.overlay_opacity, 0)
            output_frames.append(blended)
        return output_frames
```
### 4. drawers/init.py
```bash
from .tactical_view_drawer import TacticalViewDrawer
```
### 5. Tích hợp vào `main.py`
```bash
from tactical_view_converter.tactical_view_converter import TacticalViewConverter
from drawers import TacticalViewDrawer
# Step 1: Init converter
converter = TacticalViewConverter(
    tactical_court_image_path='assets/tactical_court.png',
    court_width_m=28,  # standard basketball court
    court_height_m=15,
    tactical_width_px=560,
    tactical_height_px=300
)
# Step 2: Tạo ma trận chuyển đổi từ keypoints
converter.generate_perspective_transform_matrix(camera_keypoints, tactical_keypoints)
# Step 3: Chuyển toạ độ cầu thủ sang tactical map
tactical_player_positions = converter.get_player_positions(all_player_boxes, camera_keypoints)
# Step 4: Hiển thị tactical court overlay lên video
drawer = TacticalViewDrawer()
output_frames = drawer.draw(video_frames, 'assets/tactical_court.png', 560, 300, tactical_keypoints)
```
### Kết quả:
- Tạo bản đồ chiến thuật bên góc video, vẽ overlay ảnh sân bóng + các keypoints xác định đúng góc nhìn.
- Giúp dễ dàng vẽ heatmap, đường chạy, tấn công/phòng thủ sau này.
### Lưu ý quan trọng:
| Yếu tố                 | Giải thích                                                                  |
| ---------------------- | --------------------------------------------------------------------------- |
| `camera_keypoints`     | Toạ độ landmark sân (trong frame) → nên dùng các điểm cố định như 4 góc sân |
| `tactical_keypoints`   | Tọa độ tương ứng trong tactical map (cố định)                               |
| `get_player_positions` | Trả về list các toạ độ cầu thủ trên tactical map để sau này vẽ chiến thuật  |

## 16: Validate và Lọc Court Keypoints
### Mục tiêu:
- Lọc bỏ các keypoint sai lệch gây lỗi khi tính ma trận homography.
- Dựa trên nguyên lý tỷ lệ khoảng cách giữa các cặp điểm cố định (ví dụ: điểm 13 ↔ 14 luôn cách nhau 5m thực tế). 
### 1. Thêm vào TacticalViewConverter class:
```bash
import numpy as np
import copy

class TacticalViewConverter:
    # ... các hàm trước ...

    def validate_key_points(self, court_keypoints, reference_indices=(13, 14), expected_real_distance=5.0, min_ratio=0.5, max_ratio=0.8):
        """
        court_keypoints: List of [x, y] positions
        reference_indices: tuple of two indices to use as reference for proportionality
        expected_real_distance: known real-world distance (meters or pixels) between reference points
        """
        valid_keypoints = copy.deepcopy(court_keypoints)

        idx_a, idx_b = reference_indices
        ref_a = np.array(court_keypoints[idx_a])
        ref_b = np.array(court_keypoints[idx_b])

        if np.all(ref_a == 0) or np.all(ref_b == 0):
            print("[Warning] Reference keypoints are invalid — skipping validation.")
            return valid_keypoints

        reference_pixel_distance = np.linalg.norm(ref_a - ref_b)
        if reference_pixel_distance == 0:
            print("[Warning] Reference distance is zero — invalid keypoints.")
            return valid_keypoints

        # Tính tỉ lệ "1 đơn vị thực tế ≈ bao nhiêu pixel"
        pixel_per_unit = reference_pixel_distance / expected_real_distance

        for i, point in enumerate(court_keypoints):
            if i in reference_indices or np.all(np.array(point) == 0):
                continue

            pt = np.array(point)
            d1 = np.linalg.norm(pt - ref_a)
            d2 = np.linalg.norm(pt - ref_b)
            avg_distance = (d1 + d2) / 2

            ratio = avg_distance / reference_pixel_distance

            if ratio < min_ratio or ratio > max_ratio:
                valid_keypoints[i] = [0, 0]  # loại bỏ
                print(f"[Validation] Keypoint {i} invalid (ratio={ratio:.2f}), set to [0, 0]")

        return valid_keypoints
```

- Giải thích: 

| Thành phần                   | Mục đích                                                 |
| ---------------------------- | -------------------------------------------------------- |
| `reference_indices=(13, 14)` | Cặp keypoint cố định trên sân dùng để làm chuẩn          |
| `expected_real_distance=5.0` | Khoảng cách thực tế (5m) giữa 2 điểm                     |
| `ratio < 0.5 or > 0.8`       | Nếu điểm quá gần/quá xa so với chuẩn → sai lệch, loại bỏ |
| `copy.deepcopy()`            | Tránh thay đổi dữ liệu gốc gây side effect               |
### 2. Tích hợp vào main.py:
```bash
# Sau khi lấy court_keypoints
raw_keypoints = court_keypoint_detector.get_court_keypoints(frames, read_from_stub=True)

# Validate
valid_keypoints = []
for frame_kp in raw_keypoints:
    validated = converter.validate_key_points(
        court_keypoints=frame_kp,
        reference_indices=(13, 14),
        expected_real_distance=5.0  # ví dụ 5m thật
    )
    valid_keypoints.append(validated)

# Dùng valid_keypoints để tính ma trận
converter.generate_perspective_transform_matrix(camera_keypoints=valid_keypoints[0], tactical_keypoints=known_tactical_positions)
```
### Kết quả:
- Các keypoint bị lệch hoặc sai số lớn sẽ bị loại khỏi phép biến đổi.
- Bảo vệ hệ thống khỏi lỗi "giật lag", biến dạng bản đồ chiến thuật hoặc sai vị trí cầu thủ.
## 17 Transform Player Positions to Tactical View
### Mục tiêu:
- Dùng các court keypoints đã được validate để tạo ma trận biến đổi phối cảnh.
- Dùng ma trận này để biến đổi các vị trí cầu thủ trong mỗi frame → sang tactical map (top-down view).
- Vẽ lại vị trí các cầu thủ lên bản đồ sân (tactical view).
### 1. Thêm vào TacticalViewConverter class:
```bash
import cv2
import numpy as np

class TacticalViewConverter:
    def __init__(self, tactical_image_shape=(940, 500), real_world_size_m=(28.0, 15.0)):
        self.tactical_width_px, self.tactical_height_px = tactical_image_shape
        self.real_width_m, self.real_height_m = real_world_size_m
        self.transform_matrix = None

    def generate_perspective_transform_matrix(self, camera_keypoints, tactical_keypoints):
        camera_pts = np.array(camera_keypoints, dtype=np.float32)
        tactical_pts = np.array(tactical_keypoints, dtype=np.float32)
        self.transform_matrix, _ = cv2.findHomography(camera_pts, tactical_pts)
        print("[INFO] Perspective matrix generated.")

    def transform_points(self, points):
        """
        Transform a list of points from video frame to tactical map.
        points: List of (x, y)
        return: List of (x, y) transformed
        """
        if self.transform_matrix is None:
            raise ValueError("Transformation matrix not computed.")
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.transform_matrix)
        return transformed.reshape(-1, 2)

    def get_player_positions(self, frames_player_positions, court_keypoints_per_frame):
        """
        frames_player_positions: list of list of (x, y) for each frame
        court_keypoints_per_frame: list of validated court keypoints for each frame
        return: transformed player positions per frame
        """
        all_transformed_positions = []
        for i, player_points in enumerate(frames_player_positions):
            self.generate_perspective_transform_matrix(
                camera_keypoints=court_keypoints_per_frame[i],
                tactical_keypoints=self.get_static_tactical_keypoints()
            )
            transformed = self.transform_points(player_points)
            all_transformed_positions.append(transformed)
        return all_transformed_positions

    def get_static_tactical_keypoints(self):
        """
        Trả về danh sách các keypoint cố định trên tactical map
        Cần đồng nhất thứ tự với court keypoints trong real view
        """
        # Ví dụ: keypoint theo tactical map chuẩn (giá trị pixel trên ảnh tactical)
        return [
            [50, 50], [890, 50], [50, 450], [890, 450],  # 4 góc sân
            # Các điểm khác nếu có
        ]
```
### 2. Giải thích logic:
| Phần                                    | Mục đích                                                    |
| --------------------------------------- | ----------------------------------------------------------- |
| `generate_perspective_transform_matrix` | Tạo ma trận biến đổi giữa camera view ↔ tactical map        |
| `transform_points`                      | Áp dụng phép biến đổi này để chuyển vị trí cầu thủ sang map |
| `get_player_positions`                  | Thực hiện toàn bộ quy trình này qua từng frame              |
| `get_static_tactical_keypoints`         | Keypoint cố định của tactical map (chuẩn hóa layout)        |
### 3. Vẽ cầu thủ lên tactical view (ví dụ trong TacticalViewDrawer):
```bash
class TacticalViewDrawer:
    def draw(self, frame, tactical_positions):
        for (x, y) in tactical_positions:
            if x == 0 and y == 0:
                continue  # skip invalid
            cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        return frame
```
### 4. Tích hợp vào main.py
```bash
# Giả sử đã có:
# - validated_keypoints: validated court keypoints per frame
# - player_positions: list of player (x, y) per frame

tactical_converter = TacticalViewConverter()
tactical_positions_per_frame = tactical_converter.get_player_positions(
    frames_player_positions=player_positions,
    court_keypoints_per_frame=validated_keypoints
)

# Trong loop vẽ từng frame:
for i, frame in enumerate(frames):
    tactical_frame = tactical_base_image.copy()
    tactical_frame = tactical_drawer.draw(tactical_frame, tactical_positions_per_frame[i])
    frame[0:500, 0:940] = tactical_frame  # overlay lên góc frame gốc
```
### Kết quả:
- Bạn sẽ có bản đồ cầu thủ di chuyển real-time trên tactical map 🔥.
- Chuẩn hóa dữ liệu → làm phân tích chiến thuật, heatmap, AI model phân tích chiến thuật dễ dàng.

## 18. Tính Toán Tốc Độ và Quãng Đường
### Mục tiêu:
- Tính quãng đường mỗi cầu thủ di chuyển giữa các frame (tactical map).
- Tính tốc độ = quãng đường / thời gian.
- Lưu kết quả và hiển thị trực tiếp trong video (tactical view).
### Cấu trúc thư mục
```bash
speed_and_distance_calculator/
│
├── __init__.py
├── speed_and_distance_calculator.py
```
### 1. `speed_and_distance_calculator.py`
```bash
import numpy as np
from utils.bbox_utils import measure_distance  # hoặc utils.geometry
import copy

class SpeedAndDistanceCalculator:
    def __init__(self, px_per_meter: float, frame_rate: float):
        self.px_per_meter = px_per_meter
        self.frame_rate = frame_rate
        self.prev_positions = {}  # player_id: (x, y)
        self.distances = {}       # frame_index: {player_id: distance}
        self.speeds = {}          # frame_index: {player_id: speed}

    def calculate_distance(self, tactical_player_positions: list, player_ids: list, frame_idx: int):
        """
        tactical_player_positions: [(x, y), ...] for this frame
        player_ids: matching IDs for each player in the same order
        frame_idx: index of current frame
        """
        frame_distances = {}
        frame_speeds = {}

        for pid, pos in zip(player_ids, tactical_player_positions):
            if pid not in self.prev_positions:
                self.prev_positions[pid] = pos
                continue

            prev_pos = self.prev_positions[pid]
            if pos == (0, 0) or prev_pos == (0, 0):  # Skip invalid
                continue

            dist_px = measure_distance(prev_pos, pos)
            dist_m = dist_px / self.px_per_meter
            speed_mps = dist_m * self.frame_rate

            frame_distances[pid] = dist_m
            frame_speeds[pid] = speed_mps

            self.prev_positions[pid] = pos  # update

        self.distances[frame_idx] = frame_distances
        self.speeds[frame_idx] = frame_speeds

        return frame_distances, frame_speeds
```
- Giải thích logic: 

| Thành phần                        | Giải thích                                             |
| --------------------------------- | ------------------------------------------------------ |
| `prev_positions`                  | Lưu vị trí cầu thủ ở frame trước để so sánh            |
| `measure_distance(p1, p2)`        | Trả về khoảng cách Euclid giữa 2 điểm (tactical pixel) |
| `dist_m = px / px_per_meter`      | Chuyển từ pixel sang mét                               |
| `speed = dist / (1 / frame_rate)` | Mỗi frame cách nhau 1/30s hoặc 1/25s tùy video         |

### 2. Tích hợp với pipeline
Trong `main.py` (sau khi có tactical_positions và player_ids):
```bash
speed_calculator = SpeedAndDistanceCalculator(px_per_meter=50, frame_rate=30)  # ví dụ

for i, frame in enumerate(frames):
    tactical_positions = tactical_positions_per_frame[i]
    player_ids = tracked_player_ids_per_frame[i]  # phải trích từ tracking

    dist, speed = speed_calculator.calculate_distance(tactical_positions, player_ids, i)
```
### 3. Hiển thị speed lên tactical map
Trong `TacticalViewDrawer`:
```bash
def draw_speed(self, frame, player_positions, player_ids, player_speeds):
    for (x, y), pid in zip(player_positions, player_ids):
        if pid not in player_speeds or (x, y) == (0, 0): continue
        speed = player_speeds[pid]
        label = f"{speed:.1f} m/s"
        cv2.putText(frame, label, (int(x) + 10, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
```

```bash
tactical_drawer.draw_speed(tactical_frame, tactical_positions, player_ids, speed_calculator.speeds[i])
```
### Output:
- Mỗi cầu thủ giờ có dòng hiển thị "3.5 m/s" ở tactical map.
- Có thể export toàn bộ distance/speed thành CSV cho phân tích chiến thuật/ML.
## 19. Thêm Argument Parser và Config
Mục tiêu: Biến main.py thành một module linh hoạt, dễ test, dễ deploy, chạy nhanh với các video/option khác nhau.
### 1. Sử dụng argparse trong main.py
```bash
import argparse

def define_parse_args():
    parser = argparse.ArgumentParser(description="Basketball Video Analysis Pipeline")

    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to the input video")
    parser.add_argument("--output_video", type=str, default="outputs/annotated_output.mp4",
                        help="Path to save the output video")
    parser.add_argument("--stub_dir", type=str, default="stubs/",
                        help="Directory to save/load stub data (intermediate results)")

    parser.add_argument("--overwrite_stubs", action="store_true",
                        help="Force re-running all processing even if stubs exist")

    return parser
```
#### 2. Sử dụng trong main():
```bash
def main():
    parser = define_parse_args()
    args = parser.parse_args()

    input_path = args.input_video
    output_path = args.output_video
    stub_path = args.stub_dir
    overwrite = args.overwrite_stubs

    # Example usage
    video_frames = read_video(input_path)
    save_video(output_path, video_frames)
```
### 3. Tạo configs/ thư mục
```bash
configs/
├── __init__.py
├── configs.py
```
- `configs.py`:
```bash
import os

# Default model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_DETECTION_MODEL = os.path.join(BASE_DIR, "../models/yolov8n.pt")
YOLO_POSE_MODEL = os.path.join(BASE_DIR, "../models/yolov8x-pose.pt")
DEFAULT_STUB_DIR = os.path.join(BASE_DIR, "../stubs/")
TACTICAL_COURT_IMAGE = os.path.join(BASE_DIR, "../assets/tactical_court.png")

# Other constants
FRAME_RATE = 30
PX_PER_METER = 50
```
- Trong `main.py`: 
```bash
from configs.configs import YOLO_DETECTION_MODEL, YOLO_POSE_MODEL, DEFAULT_STUB_DIR
```
## 20. Final Code Assembly & Testing
### Chuỗi pipeline tổng thể trong main.py:
```bash
def main():
  # === 1. Parse Arguments & Configs ===
  parser = define_parse_args()
  args = parser.parse_args()

  input_path = args.input_video
  output_path = args.output_video
  stub_dir = args.stub_dir
  overwrite = args.overwrite_stubs

  # === 2. Read Video ===
  frames = read_video(input_path)  # list of BGR frames

  # === 3. Detect & Track Players ===
  player_detector = PlayerDetector(...)
  player_tracks = player_detector.detect_players(frames, stub_dir, overwrite)

  # === 4. Detect & Track Ball ===
  ball_tracker = BallTracker(...)
  ball_positions = ball_tracker.track_ball(frames, stub_dir, overwrite)

  # === 5. Assign Teams ===
  team_assigner = TeamAssigner(...)
  team_data = team_assigner.assign_teams(player_tracks, stub_dir, overwrite)

  # === 6. Calculate Ball Acquisition ===
  ball_acquisition = BallAcquisitionDetector(...)
  ball_possession = ball_acquisition.detect_ball_possession(
      frames, player_tracks, ball_positions, stub_dir, overwrite
  )

  # === 7. Team Ball Control (Statistical) ===
  team_ball_control = ball_acquisition.get_team_ball_control(team_data, ball_possession)

  # === 8. Detect Court Keypoints ===
  court_detector = CourtKeypointDetector(...)
  court_keypoints = court_detector.get_court_keypoints(frames, stub_dir, overwrite)

  # === 9. Validate Court Keypoints ===
  court_transformer = TacticalViewConverter(...)
  valid_keypoints = court_transformer.validate_keypoints(court_keypoints)

  # === 10. Transform Player Positions to Tactical View ===
  tactical_player_positions = court_transformer.get_player_positions(
      player_tracks, valid_keypoints, stub_dir, overwrite
  )

  # === 11. Calculate Speed & Distance ===
  speed_distance_calc = SpeedAndDistanceCalculator(...)
  player_speeds = speed_distance_calc.calculate_distance(tactical_player_positions, stub_dir, overwrite)

  # === 12. Draw All Annotations ===
  drawer = MainDrawer(...)  # Combines all drawing tools
  annotated_frames = drawer.draw_all(
      frames=frames,
      player_tracks=player_tracks,
      ball_positions=ball_positions,
      team_data=team_data,
      ball_possession=ball_possession,
      team_ball_control=team_ball_control,
      court_keypoints=valid_keypoints,
      tactical_player_positions=tactical_player_positions,
      player_speeds=player_speeds,
      court_overlay_path=TACTICAL_COURT_IMAGE
  )

  # === 13. Save Final Video ===
  save_video(output_path, annotated_frames)
```
### Những điểm quan trọng cần test kỹ:
```bash
| Thành phần                         | Test gì?                                           | Stub? |
| ---------------------------------- | -------------------------------------------------- | ----- |
| `read_video()`                     | Đọc đầy đủ, không mất khung hình                   | ❌     |
| `player_detector.detect_players()` | Bounding box + ID ổn định                          | ✅     |
| `ball_tracker.track_ball()`        | Ball center chính xác                              | ✅     |
| `assign_teams()`                   | Mỗi player đúng team                               | ✅     |
| `detect_ball_possession()`         | Không bị nhấp nháy, đúng logic khoảng cách         | ✅     |
| `court_keypoints`                  | Các điểm hợp lý, đúng vị trí sân                   | ✅     |
| `validate_keypoints()`             | Loại bỏ keypoint sai lệch rõ ràng                  | ✅     |
| `transform_points()`               | Vị trí chuyển sang tactical map không lệch nhiều   | ✅     |
| `calculate_speed()`                | Player không dịch chuyển → speed gần 0             | ✅     |
| `draw_all()`                       | Overlay đầy đủ: track, ball, team, tactical, stats | ❌     |
```