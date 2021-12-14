# Runs the motivation experiement - retraining run and then pretrained run
import json
import os

from datetime import datetime

from ekya.CONFIG import RANDOM_SEED
from ekya.drivers.motivation.parser import get_parser
from ekya.drivers.motivation.pretrained_run import pretrained_run
from ekya.drivers.motivation.profiling_run import profiling_run
from ekya.drivers.motivation.retraining_run import retraining_run

# Set random seed for reproducibility
from ekya.utils.helpers import seed_all

seed_all(RANDOM_SEED)

if __name__ == '__main__':
    base_args = vars(get_parser().parse_args())

    all_cities = ['stuttgart', 'darmstadt', 'dusseldorf', 'monchengladbach', 'aachen', 'tubingen', 'bochum', 'bremen', 'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt', 'zurich']
    #cities = ['zurich', 'stuttgart', 'darmstadt', 'dusseldorf', 'monchengladbach', 'aachen', 'tubingen', 'bochum', 'bremen',
    #          'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt']
    # cities = ['bremen', 'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt']

    # These are the cities used in sigcomm:
    # cities = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'monchengladbach', 'stuttgart', 'tubingen', 'zurich']
    # cities = ["zurich", "jena", "cologne"]
    cities = [#'aachen', 'bochum', 'bremen',    # done on 21 June.
              #'darmstadt', # Done on 22 Jun
              'dusseldorf', 'monchengladbach', 'stuttgart',
              'tubingen']

    # Create results dir
    log_dir_name = datetime.now().strftime("%Y%m%d_%H%M")
    base_args["results_path"] = os.path.join(base_args["results_path"], log_dir_name)
    print("Creating results dir at {}".format(base_args["results_path"]))

    for city in cities:
        print("Running city {}".format(city))
        retraining_args = base_args.copy()
        retraining_args["lists_train"] = city
        retraining_args["results_path"] = os.path.join(base_args["results_path"], city)
        os.makedirs(retraining_args["results_path"], exist_ok=True)

        # Run retraining
        profiling_run(retraining_args)

    # Write experiment metadata
    metadata_path = os.path.join(base_args["results_path"], "expt_metadata.json")
    with open(metadata_path, 'w') as fp:
        json.dump(base_args, fp)

    print("Saved results at {}".format(base_args["results_path"]))