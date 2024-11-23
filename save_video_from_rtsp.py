import cv2

# URL của luồng RTSP
rtsp_url = 'rtsp://admin:Admin123@qmh1.cameraddns.net:8102/ISAPI/Streaming/Channels/302'

# Mở luồng video từ RTSP
cap = cv2.VideoCapture(rtsp_url)

# Kiểm tra xem luồng có mở thành công không
if not cap.isOpened():
    print("Không thể kết nối tới luồng video")
    exit()

# Đặt codec và tạo VideoWriter để ghi video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec XVID
output_file = 'data/video_data/output_video2.avi'  # Tên file đầu ra
fps = 20.0                                # Số khung hình trên giây
frame_size = (int(cap.get(3)), int(cap.get(4)))  # Kích thước khung hình

# Tạo VideoWriter để ghi video
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# Ghi lại video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ luồng video")
        break

    # Ghi khung hình vào file
    out.write(frame)

    # Hiển thị video
    cv2.imshow('Video Stream', frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng các tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
