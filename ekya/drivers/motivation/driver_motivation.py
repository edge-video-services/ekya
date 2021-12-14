# Runs the motivation experiement - retraining run and then pretrained run
import json
import os

from datetime import datetime
from ekya.drivers.motivation.parser import get_parser
from ekya.drivers.motivation.pretrained_run import pretrained_run
from ekya.drivers.motivation.retraining_run import retraining_run
from ekya.utils.helpers import seed_all

seed_all(42)
if __name__ == '__main__':
    base_args = vars(get_parser().parse_args())

    #cities = ['zurich', 'stuttgart', 'darmstadt', 'dusseldorf', 'monchengladbach', 'aachen', 'tubingen', 'bochum', 'bremen', 'cologne', 'ulm', 'jena', 'strasbourg', 'hamburg', 'krefeld', 'weimar', 'hanover', 'erfurt']
    cities = ['jena']

    # Create results dir
    log_dir_name = datetime.now().strftime("%Y%m%d_%H%M")
    base_args["results_path"] = os.path.join(base_args["results_path"], log_dir_name)
    print("Creating results dir at {}".format(base_args["results_path"]))
    os.makedirs(base_args["results_path"], exist_ok=True)

    # Write experiment metadata
    metadata_path = os.path.join(base_args["results_path"], "expt_metadata.json")
    with open(metadata_path, 'w') as fp:
        json.dump(base_args, fp)

    for city in cities:
        print("Running city {}".format(city))
        retraining_args = base_args.copy()
        retraining_args["lists_train"] = city
        filename = "{}_retraining_result.json".format(city)
        retraining_args["results_path"] = os.path.join(base_args["results_path"], filename)

        # Run retraining
        retraining_run(retraining_args)


        pretrained_args = base_args.copy()
        pretrained_args["lists_val"] = city
        filename = "{}_pretrained_result.json".format(city)
        pretrained_args["results_path"] = os.path.join(base_args["results_path"], filename)

        # Run pretrained model
        pretrained_run(pretrained_args)

    # Write experiment metadata
    metadata_path = os.path.join(base_args["results_path"], "expt_metadata.json")
    with open(metadata_path, 'w') as fp:
        json.dump(base_args, fp)

    print("Saved results at {}".format(base_args["results_path"]))