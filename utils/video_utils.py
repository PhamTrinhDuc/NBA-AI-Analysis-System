import cv2
import os

def read_video(video_path: str) -> list: 
  """
  Read all frames from a video file into memory

  Args: 
    video_path (str): Path to the input video file.

  Returns: 
    frames (list): list of video frames as numy arrays
  """
  cap = cv2.VideoCapture(video_path)
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret: 
      break
    frames.append(frame)
  cap.release()
  return frames

def write_video(frames: list, save_path: str, fps=30):
  """
  Save a sequence of frames as a video file
  Creates necessary directories if they don't exist and writes frames using XVID codec.
  
  Args: 
    frames (list): list of frames to save
    save_path (str): path where the video should be saved
  """
  if not frames:
    raise ValueError("No frames to save")

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  
  height, width, _ = frames[0].shape
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI format
  out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

  for frame in frames:
    out.write(frame)
  out.release()