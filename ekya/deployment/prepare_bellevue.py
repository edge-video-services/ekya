import argparse
import glob
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser("Prepare Bellevue dataset",
                                     description="Ekya script.")

    parser.add_argument("--video_dir",  type=str,
                        help="Directory where all mp4 videos sit.")
    parser.add_argument("--target_fps",  type=float, default=0.5,
                        help="Target fps the video will be encoded.")
    parser.add_argument("--save_dir",  type=str, help="Directory where "
                        "processed mp4 videos will be saved.")
    args = parser.parse_args()

    return args


args = parse_args()

assert os.path.exists(args.video_dir), f"{args.video_dir} does not exist!"
vid_paths = glob.glob(os.path.join(args.video_dir, "*.mp4"))

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


for vid_path in vid_paths:
    vid_name = os.path.basename(vid_path)
    output_vid_path = os.path.join(args.save_dir, vid_name)

    cmd = "ffmpeg -i {} -r {} {} -hide_banner".format(
        vid_path, args.target_fps, output_vid_path)
    subprocess.run(cmd.split(" "), check=True)
