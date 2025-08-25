import cv2
import os

output_dir = 'owx'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


video_path = 'D:\\ml projects\\H2_27_V_CR_NVR3_NVR3_20250728153104_20250728154205_25448729 (online-video-cutter.com).mp4'  # Replace with the path to your video file
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
while success:
  
  cv2.imwrite(os.path.join(output_dir, f"frame{count}.jpg"), image)
  success, image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1