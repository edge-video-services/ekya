# Runs the motivation experiement - retraining run and then pretrained run
import csv
import datetime
import os

from ekya.CONFIG import RANDOM_SEED
from ekya.drivers.motivation.parser import get_parser
import profiling_run

# Set random seed for reproducibility
from ekya.utils.helpers import seed_all

seed_all(RANDOM_SEED)

if __name__ == '__main__':
    GPU_ALLOC = os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]
    parser = get_parser()
    parser.add_argument("-hpid", "--hyperparameter-id", default="0", type=str,
                        help="Hyperparameter id to run.")
    # The other important arg is num_epochs, already in the parser.
    base_args = vars(parser.parse_args())

    all_cities = ['stuttgart', 'darmstadt', 'dusseldorf', 'monchengladbach', 'aachen', 'tubingen', 'bochum', 'bremen', 'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt', 'zurich']
    #cities = ['zurich', 'stuttgart', 'darmstadt', 'dusseldorf', 'monchengladbach', 'aachen', 'tubingen', 'bochum', 'bremen',
    #          'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt']
    # cities = ['bremen', 'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt']

    # These are the cities used in sigcomm:
    # cities = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'monchengladbach', 'stuttgart', 'tubingen', 'zurich']
    # cities = ["zurich", "jena", "cologne"]
    city = base_args["lists_train"]

    # Create results dir
    base_args["results_path"] = os.path.join(base_args["results_path"])
    print("Creating results dir at {}".format(base_args["results_path"]))

    print("Running city {}".format(city))
    retraining_args = base_args.copy()
    retraining_args["lists_train"] = city
    retraining_args["results_path"] = os.path.join(base_args["results_path"], str(GPU_ALLOC), city)
    os.makedirs(retraining_args["results_path"], exist_ok=True)

    # Run retraining
    results, profiling_results, e2e_times = profiling_run.profiling_run(retraining_args)

    combined_log_path = os.path.join(base_args["results_path"], 'log.csv')
    with open(combined_log_path, 'a') as f:
        wr = csv.writer(f)
        wr.writerows(e2e_times)


    # Write experiment metadata
    # metadata_path = os.path.join(base_args["results_path"], "expt_metadata.json")
    # with open(metadata_path, 'w') as fp:
    #     json.dump(base_args, fp)
    #
    # print("Saved results at {}".format(base_args["results_path"]))