import argparse
import os

from ekya.datasets.CityscapesClassification import CityscapesClassification


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare sample lists for Cityscapes dataset")
    parser.add_argument("--root", type=str, required=True,
                        help='Directory which has the leftImg8bit and gtfine dirs')
    parser.add_argument("--save-dir", type=str, required=False, default=None,
                        help='Path to generate samplelists at. Defaults to '
                             '<root>/sample_lists/citywise/')
    return parser.parse_args()


def main():
    args = parse_args()
    split = "train"
    mode = "fine"
    images_dir = os.path.join(args.root, 'leftImg8bit', split)
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(args.root, 'sample_lists/citywise/')
    os.makedirs(save_dir, exist_ok=True)
    cities = os.listdir(images_dir)
    for city in cities:
        print(f"Generating samples for city {city}")
        sample_list = CityscapesClassification.generate_sample_list(root=args.root, mode=mode,
                                                                    splits=[split], subset_cities=[city], min_res=50)
        out_file_path = os.path.join(save_dir, '{}_{}.csv'.format(city, mode))
        CityscapesClassification.dump_to_csv(sample_list[split], out_file_path)

if __name__ == '__main__':
    main()
