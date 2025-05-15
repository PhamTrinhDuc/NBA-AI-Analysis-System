from utils import read_video, write_video

def test_utils():
  input_video = "data/input_video/video_1.mp4"
  output_video = "data/output_video/video_1.avi"

  frames = read_video(video_path=input_video)
  write_video(frames=frames, save_path=output_video)
