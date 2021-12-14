import subprocess
import time

COUNT = 100
SLEEP_DUR = 5
if __name__ == '__main__':
    procs = []
    for i in range(0, 10):
        cmd = "python inference_noray.py --id {} --count {}".format(i, COUNT)
        procs.append(subprocess.Popen(cmd.split(" ")))
        time.sleep(SLEEP_DUR)

    for proc in procs:
        proc.wait()