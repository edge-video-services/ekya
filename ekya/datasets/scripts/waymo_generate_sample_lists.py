import argparse
import copy
import os

import pandas as pd

from ekya.datasets.WaymoClassification import WaymoClassification


NUM_SEGS = 10
OFFSET = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare sample lists from Waymo Open Dataset")
    parser.add_argument("--root", type=str, required=True,
                        help='Directory which stores all segments(.tfrecord)')
    parser.add_argument("--save-dir", type=str, required=True,
                        help='Path to results.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    sorted_segs = WaymoClassification.sort_segments(args.root, args.save_dir)
    for location in sorted_segs:
        city = location.split('_')[1]
        img_save_dir = os.path.join(
            args.save_dir, 'waymo_classification_images')
        file_save_dir = os.path.join(
            args.save_dir, 'waymo_classification_images', 'sample_lists',
            'citywise')
        os.makedirs(file_save_dir, exist_ok=True)

        WaymoClassification.generate_sample_list(
            os.path.join(args.save_dir, 'sorted_segments.json'), city,
            img_save_path=img_save_dir, write_filename=os.path.join(
                file_save_dir, '{}_labels.csv'.format(city)))

        # Split a large sample list into many short sample lists.
        sample_list_file = os.path.join(file_save_dir, f'{city}_labels.csv')
        samples = pd.read_csv(sample_list_file)
        assert isinstance(samples, pd.DataFrame)
        sorted_segs = samples['segment'].unique()

        seg_win = []
        for i, seg in enumerate(sorted_segs[OFFSET:OFFSET + NUM_SEGS]):
            if (i + 1) % NUM_SEGS == 0:
                print(i, i - NUM_SEGS + 1, seg_win)
                sample2save = copy.deepcopy(samples[samples['segment'].isin(seg_win)])
                sample2save["idx"] = pd.Series(
                    range(0, len(sample2save["idx"]))).values
                sample2save.set_index("idx", inplace=True)
                print(sample2save['segment'].unique())
                outfile = os.path.join(
                    file_save_dir, '{}_{:03d}_{:03d}_labels.csv'.format(
                    city, i - NUM_SEGS+1 + OFFSET, i + OFFSET))
                sample2save.to_csv(outfile)

                seg_win = []
            seg_win.append(seg)


if __name__ == '__main__':
    main()
