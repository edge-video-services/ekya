# Runs the motivation experiement - cont retraining vs pretrained vs trained once every 5 tasks.
import json
import os

from datetime import datetime
from ekya.drivers.motivation.parser import get_parser
from ekya.drivers.motivation.retrainingfreq_run import retrainingfreq_run

if __name__ == '__main__':
    base_args = vars(get_parser().parse_args())

    all_cities = ['stuttgart', 'darmstadt', 'dusseldorf', 'monchengladbach', 'aachen', 'tubingen', 'bochum', 'bremen', 'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt', 'zurich']
    cities = ['jena']
    #cities = ['zurich', 'bremen', 'aachen', 'hanover', 'bochum']

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
        retrainingfreq_run(retraining_args)

    # Write experiment metadata
    metadata_path = os.path.join(base_args["results_path"], "expt_metadata.json")
    with open(metadata_path, 'w') as fp:
        json.dump(base_args, fp)

    print("Saved results at {}".format(base_args["results_path"]))