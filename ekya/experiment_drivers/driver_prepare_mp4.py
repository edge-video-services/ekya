import argparse
import glob
import os

from ekya.datasets.Mp4VideoClassification import Mp4VideoClassification


def parse_args():
    parser = argparse.ArgumentParser("Prepare MP4(Bellevue, Vegas) dataset",
                                     description="Ekya script.")
    parser.add_argument("--dataset", type=str,
                        required=True, help="Dataset name.")
    parser.add_argument('--model-path', type=str, required=True,
                        help='Object detection model path.')
    parser.add_argument('--device', type=int, required=True,
                        help='GPU device number.')
    parser.add_argument("--dataset-root",  type=str, required=True,
                        help="Directory where folders named as dataset names "
                        "sit")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    dataset_name = args.dataset
    root = os.path.join(args.dataset_root, dataset_name)
    vid_paths = glob.glob(os.path.join(root,  '*.mp4'))

    for vid_path in sorted(vid_paths):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        # create frame image folder
        img_dir = os.path.join(root, vid_name, 'frame_images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # create classificaiton image folder
        clf_img_dir = os.path.join(root, vid_name, 'classification_images')
        if not os.path.exists(clf_img_dir):
            os.makedirs(clf_img_dir)

        # extract frames images
        Mp4VideoClassification.extract_frame_images(vid_path, img_dir)

        # generate object detection
        Mp4VideoClassification.generate_object_detection(
            vid_path, img_dir, os.path.join(root, vid_name),
            args.model_path, args.device)

        Mp4VideoClassification.generate_sample_list_from_detection_file(
            vid_path,
            os.path.join(root, vid_name, f'{vid_name}_detections.csv'),
            os.path.join(root, 'sample_lists', 'citywise'), min_res=30)


if __name__ == '__main__':
    main()
