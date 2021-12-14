import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Ekya simulator")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to Ekya profiles.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=['cityscapes', 'waymo', 'mp4'],
                        help='Dataset name.')
    parser.add_argument("--camera_names", type=str,  nargs='*', required=True,
                        help="A list of cameras/cities to be used.")
    parser.add_argument("--retraining_periods", nargs='*', type=int,
                        default=[100], help="A list of retraining periods "
                        "(second) to be tested")
    parser.add_argument("--delay", type=float, default=0, help="Golden model "
                        "delay (second) costed to generate ground truth "
                        "labels.")
    parser.add_argument("--provisioned_resources", nargs='*', type=float,
                        default=[8, 4, 2, 1], help="A list of gpu numbers to"
                        "be provided in the simulator.")
    parser.add_argument("--hyp_map_path", type=str, required=True, help="Path"
                        "to hyperparameter map.")
    parser.add_argument("--hyperparameters", nargs='*', type=int,
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                 14, 15, 16, 17], help="A list of "
                        "hyperparameter ids provided to the simulator. We"
                        "18 hyperparameters for retraining.")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help='Number of tasks/retrainining windows')
    parser.add_argument("--real_inference_profiles", type=str,
                        required=True, help='Path to real inference profiles.')
    parser.add_argument("--iterations", type=int, default=3, help='Number of '
                        'iterations used in the thief scheduler.')
    parser.add_argument("--output_path", type=str, required=True, help="Path"
                        "to store the output files of the simulation.")

    # Fair fixed scheduler args
    parser.add_argument("--fairfixed_config_id", type=str, required=False, help="Fair fixed scheduler config id to use")
    parser.add_argument("--fairfixed_config_epochs", type=str, required=False, help="Fair fixed scheduler config epochs to use")

    # Cloud scheduler args
    parser.add_argument("--cloud_delay", type=str, required=False, default="", help="csv of cloud delays to test. Eg. 100,200,300")

    args = parser.parse_args()
    return args
