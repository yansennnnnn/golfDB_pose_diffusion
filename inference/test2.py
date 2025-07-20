import os
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from PIL import Image

# 要提取的帧编号（从1开始计数）
target_frames = [5, 15, 30]

def extract_frames(video_path, frame_indices):
    clip = VideoFileClip(video_path)
    video_fps = clip.fps
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 创建输出文件夹
    output_folder = f"{video_name}_frames"
    os.makedirs(output_folder, exist_ok=True)

    for idx in frame_indices:
        # 注意：MoviePy 使用秒为单位，所以换算为时间戳
        t = (idx - 1) / video_fps
        if t > clip.duration:
            print(f"[跳过] 第 {idx} 帧超出视频 {video_name} 的时长")
            continue

        frame = clip.get_frame(t)  # 获取指定时间的帧（返回的是numpy array）
        img = Image.fromarray(frame)
        img.save(os.path.join(output_folder, f"frame_{idx}.jpg"))
        print(f"[保存] {video_name}: frame {idx} @ {t:.2f}s")

    clip.close()

def main():
    for filename in os.listdir("."):
        if filename.lower().endswith(".mp4"):
            extract_frames(filename, target_frames)

if __name__ == "__main__":
    main()
