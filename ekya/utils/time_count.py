# Script to count total time of videos in current dir.
import glob
from subprocess import PIPE, Popen

command = 'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '
total_time = 0
for f in glob.glob("*.mp4"):
    with Popen(command + f, stdout=PIPE, stderr=None, shell=True) as process:
        output = process.communicate()[0].decode("utf-8")
        seconds = float(output.strip())
        total_time+=seconds
print(f"Total time in seconds: {total_time}. That's {total_time/60} minutes, or {total_time/3600} hours.")