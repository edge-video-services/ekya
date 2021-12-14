import os
import time

from ekya.models.resnet_inference import ResnetInference
import argparse

LOG_BASE_PATH = "./tmp/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="id")
    parser.add_argument("--count", type=int, help="number of runs to run")
    parser.add_argument("--cuda", help="use cuda", action="store_true")
    args = parser.parse_args()

    model = ResnetInference()
    imgs = ["image.jpg"]*args.count
    os.makedirs(LOG_BASE_PATH, exist_ok=True)
    log_file = open(os.path.join(LOG_BASE_PATH, "{}.csv".format(args.id)), 'w')
    for img_path in imgs:
        start_time = time.time()
        #result = model.infer_image(img_path)
        result = time.sleep(1)
        end_time = time.time()
        log_file.write(",".join(map(str, [time.time(), end_time-start_time])) + "\n")
        log_file.flush()
    print(result)