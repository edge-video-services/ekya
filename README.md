# Ekya: Continuous Learning on the Edge
Ekya is a system which enables continuous learning on resource constrained devices.
Given a set of video streams and pre-trained models, Ekya can continuously fine-tune the models to maximize accuracy by intelligently allocating resources between live inference and retraining in the background. 

At the core of Ekya is the Thief Scheduler, which operates by stealing small resource chunks from a selected job and reallocating them to a more promising job. The thief scheduler obtains information about the "promise" of a job through the micro-profiling mechanism, which runs each retraining job for a short duration to estimate it's future performance.

This architecture diagram highlights the flow of data in Ekya. More details can be found in our NSDI 2022 paper available [here](https://github.com/edge-video-services/ekya/raw/main/assets/Ekya_nsdi22_camready.pdf).

<p align="center">
    <img src="https://i.imgur.com/ng1jLsS.png" width="500">
</p>

With this release of Ekya, you can: 
* :white_check_mark: **Code** - Run Ekya on four supported datasets (Cityscapes, Waymo, UrbanTraffic, UrbanBuilding), or your own custom MP4 files (see [Running Ekya](#running-ekya)). 
* :white_check_mark: **Datasets** - Use our newly released UrbanTraffic and UrbanBuilding datasets for your own research (see [New Datasets](#new-datasets)).
* :white_check_mark: **Extend Ekya** - Build upon Ekya to build and prototype new scheduling algorithms and continuous learning techniques (see [Extending Ekya](#extending-ekya)). 

[](mdtoc)
# Table of Contents

* [New Datasets](#new-datasets)
	* [Urban Traffic Dataset](#urban-traffic-dataset)
	* [Urban Building Dataset](#urban-building-dataset)
* [Running Ekya](#running-ekya)
	* [Installation](#installation)
	* [Preparing Models](#preparing-models)
		* [Golden Model](#golden-model)
		* [Object Detection Model](#object-detection-model)
	* [Running Ekya with Cityscapes Dataset](#running-ekya-with-cityscapes-dataset)
		* [Preprocessing the Cityscapes Dataset](#preprocessing-the-cityscapes-dataset)
		* [Running Ekya](#running-ekya-1)
	* [Running Ekya with the Waymo Dataset](#running-ekya-with-the-waymo-dataset)
		* [Download pretrained models](#download-pretrained-models)
		* [Download Preprocessed Waymo Dataset](#download-preprocessed-waymo-dataset)
		* [Regenerate processed Waymo Dataset from Original Waymo Dataset](#regenerate-processed-waymo-dataset-from-original-waymo-dataset)
		* [Running Waymo](#running-waymo)
	* [Running Ekya with Urban Traffic Dataset](#running-ekya-with-urban-traffic-dataset)
		* [Download Pretrained Models](#download-pretrained-models-1)
		* [Prepare Urban Traffic Dataset from mp4 Videos](#prepare-urban-traffic-dataset-from-mp4-videos)
		* [Running Urban Traffic Dataset](#running-urban-traffic-dataset)
	* [Running Ekya with Urban Building Dataset](#running-ekya-with-urban-building-dataset)
		* [Download Pretrained Models](#download-pretrained-models-2)
		* [Download Preprocessed Urban Building Dataset](#download-preprocessed-urban-building-dataset)
		* [Prepare Urban Building Dataset from mp4 Videos](#prepare-urban-building-dataset-from-mp4-videos)
		* [Running Urban Building Dataset](#running-urban-building-dataset)
	* [Plotting Results](#plotting-results)
* [Comparing against other baselines](#comparing-against-other-baselines)
* [Extending Ekya](#extending-ekya)
	* [Adding Custom Schedulers to Ekya](#adding-custom-schedulers-to-ekya)
	* [Adding Custom Learning Techniques to Ekya](#adding-custom-learning-techniques-to-ekya)
* [Frequently Asked Questions](#frequently-asked-questions)
* [Ekya driver script usage guide](#ekya-driver-script-usage-guide)
* [Citing Ekya](#citing-ekya)
[](/mdtoc)


# New Datasets
As a part of this repository, we present two new video datasets - Urban Traffic and Urban Building. In addition, Ekya can also run on the Cityscapes and Waymo datasets (see instructions below).

We have labelled both Urban Traffic and Urban Building datasets using our golden model (ResNeXT-101 trained on MS COCO). These labels are stored in files called samplelists. 
The samplelists for each video clip, containing the objects detected and their labels can be found in the `samplelists` directory in the dataset folder. 

Each samplelist is a CSV with 6 columns: `["idx", "class", "x0", "y0", "x1", "y1"]`. Each column is described below:

* `idx`: row index
* `class`: Object class - mapping can be found in `/ekya/datasets/coco_classes.txt`.
* `x0`: X coordinates of top left of bounding box
* `y0`: Y coordinates of top left of bounding box
* `x1`: X coordinates of bottom right of bounding box
* `y1`: Y coordinates of bottom right of bounding box

Origin for the image is at the top right of the video frame.   

## Urban Traffic Dataset
<p align="center">
    <img src="https://i.imgur.com/tV4M1oZ.png" width="500">
</p>

This dataset contains 62GB of traffic videos recorded from five pole mounted fish-eye cameras in the city of Bellevue, WA. Each video stream is recorded at 1280x720@30fps, for a total of 101 hour of video across all cameras. 

Download links:
* [Camera 1 Videos](https://drive.google.com/drive/folders/16coOR8PlNzvmUm1vsaYJVF_bAOQGySa8?usp=sharing)
* [Camera 2 Videos](https://drive.google.com/drive/folders/1cR1VwoAvEjFLRaUzeYph-bxx4LoM6pOH?usp=sharing)
* [Camera 3 Videos](https://drive.google.com/drive/folders/1irB6XKu2iM3BSJ2AEYH4kJl9nfG9j-yy?usp=sharing)
* [Camera 4 Videos](https://drive.google.com/drive/folders/1IN6kwywddO3B3uHyC5S18vqf0KEWToJ_?usp=sharing)
* [Camera 5 Videos](https://drive.google.com/drive/folders/17bn7l7Qm5s-r5DYoFQPhviFZ0jWY9qk5?usp=sharing)
* [Combined labels and cropped objects](https://drive.google.com/drive/folders/177UvUO26lDybXGzyy_8QL9ov1dRyFfhi?usp=sharing)
 

## Urban Building Dataset
<p align="center">
    <img src="https://i.imgur.com/eMK2C9f.jpg" width="500">
</p>

This dataset contains 24 hours of video recorded from a PTZ public camera with a non-stationary view in Las Vegas. The video is recorded at 1920x1080@0.2fps. Along with the video stream, we provide the labels in the samplelist format described above.

The dataset, object labels and cropped images of objects can be [downloaded here](https://drive.google.com/drive/folders/1wuAVAQQ4rfhg7rIsFIYB2IG32y0r3AYG?usp=sharing). 

# Running Ekya
## Installation

1. Checkout Ray repository. Ekya requires first building a particular branch of Ray from source. Ekya uses commit `cf53b351471716e7bfa71d36368ebea9b0e219c5` (`Ray 0.9.0.dev0`) from the Ray repository.
`pip install ray` is not sufficient.
```bash
git clone https://github.com/ray-project/ray/
cd ray
git checkout cf53b35
```
2. To build Ray, follow the [build instructions](https://docs.ray.io/en/master/development.html#building-ray-on-linux-macos-full) from the Ray repository.
```
sudo apt-get update
sudo apt-get install -y build-essential curl unzip psmisc ffmpeg

# Install Cython
pip install cython==0.29.0 pytest

# Install Bazel.
ray/ci/travis/install-bazel.sh
# (Windows users: please manually place Bazel in your PATH, and point
# BAZEL_SH to MSYS2's Bash: ``set BAZEL_SH=C:\Program Files\Git\bin\bash.exe``)

# Build the dashboard
# (requires Node.js, see https://nodejs.org/ for more information).
# If folder "ray/dasboard/client" does not exist, please move forward to
# "Install ray"
pushd ray/dashboard/client
npm install
npm run build
popd

# Install Ray.
cd ray/python
pip install -e . --verbose  # Add --user if you see a permission denied error.
```
3. After installing ray, clone the Ekya repository and install Ekya.
```
git clone https://github.com/edge-video-services/ekya/
pip install -e . --verbose
```
4. Install [Nvidia Multiprocess Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html).
```
sudo apt-get update
sudo apt-get install nvidia-cuda-mps
```
5. Set your GPU to run in exclusive process mode and run Nvidia MPS daemon. This will require killing Xserver if it is running.
```
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
```

**NOTE**: Starting version `410.74`, Nvidia MPS does not necessarily honor GPU resource allocation for tasks. Please use version `392` or lower.  

## Preparing Models

### Golden Model

The golden model is used to generate image classification groundtruth in Ekya.
Please download resnext101 elastic model from
[here](https://github.com/allenai/elastic) into ```ekya/golden_model/```.


### Object Detection Model

The object detection model is used to identify objects from video frames.
Please download ```faster_rcnn_resnet101_coco_2018_01_28``` from
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
into ```ekya/object_detection_model/```.

## Running Ekya with Cityscapes Dataset

### Preprocessing the Cityscapes Dataset
1. Download the Cityscapes dataset using the instructions on the [website](https://www.cityscapes-dataset.com/) and extract the `leftImg8bit` and `gtFine` subdirectorie to a dataset directory.
2. Generate the samplelists by running
```bash
cd ekya/datasets/scripts
python cityscapes_generate_sample_lists.py --root <path to your cityscapes root>
```

### Running Ekya
3. Download the pretrained models for citysapes from [here](https://drive.google.com/drive/folders/15qE5IBFAkKuiDeUcV8xQPvpKXq1Zk6yT?usp=sharing) and extract them to a directory.
4. Run the multicity training script provided with Ekya.
 ```
./ekya/experiment_drivers/driver_multicity.sh
```
You may need to modify `DATASET_PATH` and `MODEL_PATH` to point to your dataset and pretrained models dir, respectively. You must also set `NUM_GPUS` to reflect the number of GPUs to use.
This script will run all schedulers, including `thief`, `fair` and `noretrain`.

5. The results will be written in a timestamped directory at `results/ekya_expts/cityscapes/`.

## Running Ekya with the Waymo Dataset


### Download pretrained models
1. Download ```waymo_pretrain_model.tar``` from
   [here](https://drive.google.com/drive/u/1/folders/1dJjnrHfV86eYB4nuMFrNU_kPUzzSknXb)
   into ```pretrained_models```.
2. Perform the following commands
    ```bash 
    cd ekya/pretrained_models
    tar -xvf waymo_pretrain_model.tar
    mv waymo_pretrain_model waymo
    rm waymo_pretrain_model.tar
    ```

### Download Preprocessed Waymo Dataset
1. Download ```waymo_classification_images.tar``` from
   [here](https://drive.google.com/drive/u/1/folders/1dJjnrHfV86eYB4nuMFrNU_kPUzzSknXb)
   into ```dataset/waymo```.
2. Perform the following commands
    ```bash 
    cd dataset/waymo
    tar -xvf waymo_classification_images.tar
    rm waymo_classification_images.tar
    ```
### Regenerate processed Waymo Dataset from Original Waymo Dataset
1. Go to [Waymo Open Dataset](https://waymo.com/intl/en_us/dataset-download-terms/).
2. Under "Perception Dataset", go to "v1.0, August 2019: Initial release".
3. Click "tar files" and download all tar files into "dataset/waymo/tfrecord"
5. Decompress all tar files.
6. Then perform the following commands.
    ```bash
    cd ekya/datasets/scripts
    python waymo_generate_sample_lists.py --root ../../../dataset/waymo/tfrecord --save-dir ../../../dataset/waymo
    ```

### Running Waymo
```bash
cd ekya/experiment_drivers
bash driver_profiling_waymo_golden.sh
```


## Running Ekya with Urban Traffic Dataset

### Download Pretrained Models
1. Download ```bellevue_pretrained_models.tar.gz``` from
[here](https://drive.google.com/drive/folders/1wuAVAQQ4rfhg7rIsFIYB2IG32y0r3AYG)
into ```pretrained_models```.
2. Perform the following commands. 
    ```bash 
    cd pretrained_models
    tar -xvf bellevue_pretrained_models.tar.gz
    mv bellevue_pretrained_models bellevue
    rm bellevue_pretrained_models.tar.gz
    ```

### Prepare Urban Traffic Dataset from mp4 Videos

```bash
cd ekya/experiment_drivers
python driver_prepare_mp4.py \
    --dataset bellevue \
    --dataset-root ../../dataset \
    --device 0 \
    --model-path ../../object_detection_model/faster_rcnn_resnet101_coco_2018_01_28
```

### Running Urban Traffic Dataset

```bash
cd ekya/experiment_drivers
bash driver_profiling_mp4_golden_vegas.sh
```


## Running Ekya with Urban Building Dataset

### Download Pretrained Models

1. Download ```vegas_pretrained_models.tar.gz``` from
[here](https://drive.google.com/drive/folders/1wuAVAQQ4rfhg7rIsFIYB2IG32y0r3AYG)
into ```pretrained_models```.
2. Run the following commands. 
    ```bash 
    cd pretrained_models
    tar -xvf vegas_pretrained_models.tar.gz
    mv vegas_pretrained_models vegas
    rm vegas_pretrained_models.tar.gz
    ```

### Download Preprocessed Urban Building Dataset
1. Download ```las_vegas_24h_[0-3].tar.gz``` and
   ```vegas_sample_lists.tar.gz``` from
   [here](https://drive.google.com/drive/folders/1wuAVAQQ4rfhg7rIsFIYB2IG32y0r3AYG)
   into ```datasets/vegas```.

2. Decompress
    ```bash
    cd datasets/vegas
    for i in {0..3}; do tar -xf las_vegas_24h_$i.tar.gz; done
    tar -xf vegas_sample_lists.tar.gz
    rm *.tar.gz
    ```

### Prepare Urban Building Dataset from mp4 Videos
1. Download ```las_vegas_24h_[0-3].mp4``` 
   [here](https://drive.google.com/drive/folders/1wuAVAQQ4rfhg7rIsFIYB2IG32y0r3AYG)
   into ```datasets/vegas```.
2. Run the following commands.
    ```bash
    cd ekya/experiment_drivers
    python driver_prepare_mp4.py \
        --dataset vegas \
        --dataset-root ../../dataset \
        --device 0 \
        --model-path ../../object_detection_model/faster_rcnn_resnet101_coco_2018_01_28
    ```

### Running Urban Building Dataset
```bash
cd ekya/experiment_drivers
bash driver_profiling_mp4_golden_bellevue.sh
```


## Plotting Results
To plot the results from the above runs,have done so, collect all the result directories. You can then use the `/viz/driver_viz_multicity_varyingcities.ipynb` notebook to plot your results. You will need to set the `BASE_DIR` to the root of your Ekya log directory. 

For example, if you run the cityscapes driver script with the default values (defaults are set for a shorter run), you should be able to produce the following figure:
  
<p align="center">
    <img src="https://i.imgur.com/ilwGMs8.png" width="300">
</p>

To create figures with varying GPU counts, you will need to run the driver script for different `NUM_GPUS` counts and collate them into one directory before using `/viz/driver_viz_multicity_varyingcities.ipynb`. 


# Comparing against other baselines

One of the baselines explored in our NSDI paper is the comparison against a continous model selection strategy.
This baseline strategy uses pre-cached models generated under different scenarios (e.g. weather, time of day, class distributions) and loads models according to the current scenario. As we demonstrate in the paper, Ekya outperforms this strategy:

<p align="center">
    <img src="https://i.imgur.com/49pxG3i.png" width="400">
</p>

To run these baselines, follow these steps:

```bash
# assume waymo dataset is ready
cd ekya/model_cache
# to train models used in the model cache experiments
bash driver_model_cache.sh
# to do the inference
bash driver.sh
# to plot figures
python plot.py
```

# Extending Ekya
Ekya can be easily extended in two dimensions - adding custom schedulers and adding new continuous learning techniques.

## Adding Custom Schedulers to Ekya
Ekya schedulers are implemented in `ekya/schedulers/`. Any new scheduler must extend the Scheduler base class in `scheduler.py`.

The `BaseScheduler` class implements two key methods - `reallocation_callback` and `get_inference_schedule`. Their method signature and usage is described below.

```python
class BaseScheduler(object):
    def reallocation_callback(self,
                              completed_camera_name: str,
                              inference_resource_weights: dict,
                              training_resources_weights: dict) -> [dict, dict]:
        '''
        This callback is called when a training job completes. This provides the scheduler an opportunity to reconfigure
        resource allocations for jobs. Currently, only changes to to the inference resources are reflected
        (because updating training jobs would require process restarts, an expensive operation).
        :param completed_camera_name: str, name of the job completed
        :param inference_resource_weights: the current inference resource allocation
        :param training_resources_weights: the current training resource allocation
        :return: new_inference_resource_weights, new_training_resources_weights, two dictionaries mapping resource weights for inference and training jobs.
        '''
        pass

    def get_inference_schedule(self,
                                cameras: List[Camera],
                                resources: float):
        '''
        Returns the schedule when inference only jobs must be run. This must be super fast since this is the schedule
        used before the get_schedule actual schedule is obtained.
        :param cameras: list of cameras
        :param resources: total resources in the system to be split across tasks
        :return: inference resource weights, hyperparameters to use inference.
        '''
        pass
```

## Adding Custom Learning Techniques to Ekya
Currently Ekya uses simple gradient updates to update vision models for each camera running in the system.
This repository also includes another incremental learning technique [ICaRL (CVPR 2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf) using this [implementation](https://github.com/arthurdouillard/incremental_learning.pytorch).

To add your own learning technique:

0. Your model must extend `ekya.classes.MLModel` baseclass. 
1. Add your model to `/ekya/classes`.
2. In `/ekya/classes/model.py`, replace `MLModel` with your Model in `RayMLModel = ray.remote(num_gpus=0.01)(<model>)`

# Frequently Asked Questions

1. When installing ray with `pip install -e . --verbose` and encountering the
   error `"[ray] [bazel] build failure, error --experimental_ui_deduplicate
   unrecognized"`.

    Please checkout this
    [issue](https://github.com/ray-project/ray/issues/11237). If other versions
    of `bazel` are installed, please install `bazel-3.2.0` following instructions
    from
    [here](https://docs.bazel.build/versions/main/install-compile-source.html)
    and compile ray using `bazel-3.2.0`.


# Ekya driver script usage guide
```
usage: Ekya  [-h] [-ld LOG_DIR] [-retp RETRAINING_PERIOD]
                    [-infc INFERENCE_CHUNKS] [-numgpus NUM_GPUS]
                    [-memgpu GPU_MEMORY] [-r ROOT]
                    [--dataset-name {cityscapes,waymo}] [-c CITIES]
                    [-lpt LISTS_PRETRAINED] [-lp LISTS_ROOT] [-dc]
                    [-ir RESIZE_RES] [-w NUM_WORKERS] [-ts TRAIN_SPLIT]
                    [-dtfs] [-hw HISTORY_WEIGHT] [-rp RESTORE_PATH]
                    [-cp CHECKPOINT_PATH] [-mn MODEL_NAME] [-nc NUM_CLASSES]
                    [-b BATCH_SIZE] [-lr LEARNING_RATE] [-mom MOMENTUM]
                    [-e EPOCHS] [-nh NUM_HIDDEN] [-dllo] [-sched SCHEDULER]
                    [-usp UTILITYSIM_SCHEDULE_PATH]
                    [-usc UTILITYSIM_SCHEDULE_KEY] [-mpd MICROPROFILE_DEVICE]
                    [-mprpt MICROPROFILE_RESOURCES_PER_TRIAL]
                    [-mpe MICROPROFILE_EPOCHS]
                    [-mpsr MICROPROFILE_SUBSAMPLE_RATE]
                    [-mpep MICROPROFILE_PROFILING_EPOCHS]
                    [-fswt FAIR_INFERENCE_WEIGHT] [-nt NUM_TASKS]
                    [-stid START_TASK] [-ttid TERMINATION_TASK]
                    [-nsp NUM_SUBPROFILES] [-op RESULTS_PATH] [-uhp HYPS_PATH]
                    [-hpid HYPERPARAMETER_ID] [-pm] [-pp PROFILE_WRITE_PATH]
                    [-ipp INFERENCE_PROFILE_PATH]
                    [-mir MAX_INFERENCE_RESOURCES]

Ekya driver script for cityscapes dataset. Uses pretrained models to improve accuracy over time.

optional arguments:
  -h, --help            show this help message and exit
  -ld LOG_DIR, --log-dir LOG_DIR
                        Directory to log results to
  -retp RETRAINING_PERIOD, --retraining-period RETRAINING_PERIOD
                        Retraining period in seconds
  -infc INFERENCE_CHUNKS, --inference-chunks INFERENCE_CHUNKS
                        Number of inference chunks per retraining window.
  -numgpus NUM_GPUS, --num-gpus NUM_GPUS
                        Number of GPUs to partition.
  -memgpu GPU_MEMORY, --gpu-memory GPU_MEMORY
                        Per GPU Memory in GB.
  -r ROOT, --root ROOT  Path to cityscapes dataset root.
  --dataset-name {cityscapes,waymo}
                        Name of the dataset supported.
  -c CITIES, --cities CITIES
                        comma separated str of list of cities to create
                        cameras. Num cameras = num of cities
  -lpt LISTS_PRETRAINED, --lists-pretrained LISTS_PRETRAINED
                        comma separated str of lists used for training the
                        pretrained model. Used as history for continuing the
                        retraining. Usually frankfurt,munster.
  -lp LISTS_ROOT, --lists-root LISTS_ROOT
                        root of sample lists. This must be downloaded from ekya repo.
  -dc, --use-data-cache
                        Use data caching for cityscapes. WARNING: Might
                        consume lot of disk space.
  -ir RESIZE_RES, --resize-res RESIZE_RES
                        Image size to use for cityscapes.
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of workers preprocessing the data.
  -ts TRAIN_SPLIT, --train-split TRAIN_SPLIT
                        Train validation split. This float is the fraction of
                        data used for training, rest goes to validation.
  -dtfs, --do-not-train-from-scratch
                        Do not train from scratch for every profiling task -
                        carry forward the previous model
  -hw HISTORY_WEIGHT, --history-weight HISTORY_WEIGHT
                        Weight to assign to historical samples when
                        retraining. Between 0-1. Cannot be zero. -1 if no
                        reweighting.
  -rp RESTORE_PATH, --restore-path RESTORE_PATH
                        Path to the pretrained models to use for init. Must be
                        downloaded from Ekya repo.
  -cp CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Path where to save the model
  -mn MODEL_NAME, --model-name MODEL_NAME
                        Model name. Can be resnetXX for now.
  -nc NUM_CLASSES, --num-classes NUM_CLASSES
                        Number of classes per task.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size.
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate.
  -mom MOMENTUM, --momentum MOMENTUM
                        Momentum.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs per task.
  -nh NUM_HIDDEN, --num-hidden NUM_HIDDEN
                        Number of neurons in hidden layer.
  -dllo, --disable-last-layer-only
                        Adjust weights on all layers, instead of modifying
                        just last layer.
  -sched SCHEDULER, --scheduler SCHEDULER
                        Scheduler to use. Either of fair, noretrain, thief,
                        utilitysim.
  -usp UTILITYSIM_SCHEDULE_PATH, --utilitysim-schedule-path UTILITYSIM_SCHEDULE_PATH
                        Path to the schedule (period allocation) generated by
                        utilitysim.
  -usc UTILITYSIM_SCHEDULE_KEY, --utilitysim-schedule-key UTILITYSIM_SCHEDULE_KEY
                        The top level key in the schedule json. Usually of the
                        format {}_{}_{}_{}.format(period,res_count,scheduler,u
                        se_oracle)
  -mpd MICROPROFILE_DEVICE, --microprofile-device MICROPROFILE_DEVICE
                        Device to microprofile on - either of cuda, cpu or
                        auto
  -mprpt MICROPROFILE_RESOURCES_PER_TRIAL, --microprofile-resources-per-trial MICROPROFILE_RESOURCES_PER_TRIAL
                        Resources required per trial in microprofiling. Reduce
                        this to run multiple jobs in together while
                        microprofiling. Warning: may cause OOM error if too
                        many run together.
  -mpe MICROPROFILE_EPOCHS, --microprofile-epochs MICROPROFILE_EPOCHS
                        Epochs to run microprofiling for.
  -mpsr MICROPROFILE_SUBSAMPLE_RATE, --microprofile-subsample-rate MICROPROFILE_SUBSAMPLE_RATE
                        Subsampling rate while microprofiling.
  -mpep MICROPROFILE_PROFILING_EPOCHS, --microprofile-profiling-epochs MICROPROFILE_PROFILING_EPOCHS
                        Epochs to generate profiles for, per hyperparameter.
  -fswt FAIR_INFERENCE_WEIGHT, --fair-inference-weight FAIR_INFERENCE_WEIGHT
                        Weight to allocate for inference in the fair
                        scheduler.
  -nt NUM_TASKS, --num-tasks NUM_TASKS
                        Number of tasks to split each dataset into
  -stid START_TASK, --start-task START_TASK
                        Task id to start at.
  -ttid TERMINATION_TASK, --termination-task TERMINATION_TASK
                        Task id to end the Ekya loop at. -1 runs all tasks.
  -nsp NUM_SUBPROFILES, --num-subprofiles NUM_SUBPROFILES
                        Number of tasks to split each dataset into
  -op RESULTS_PATH, --results-path RESULTS_PATH
                        The josn file to write results to.
  -uhp HYPS_PATH, --hyps-path HYPS_PATH
                        hyp_map.json path which lists the hyperparameter_id to
                        hyperparameter mapping.
  -hpid HYPERPARAMETER_ID, --hyperparameter-id HYPERPARAMETER_ID
                        Hyperparameter id to use for retraining. From hyps-
                        path json.
  -pm, --profiling-mode
                        Run in profiling mode?
  -pp PROFILE_WRITE_PATH, --profile-write-path PROFILE_WRITE_PATH
                        Run in profiling mode?
  -ipp INFERENCE_PROFILE_PATH, --inference-profile-path INFERENCE_PROFILE_PATH
                        Path to the inference profiles csv
  -mir MAX_INFERENCE_RESOURCES, --max-inference-resources MAX_INFERENCE_RESOURCES
                        Maximum resources required for inference. Acts as a
                        ceiling for the inference scaling function.

```

# Citing Ekya
If you use Ekya or the new datasets in your research, please cite the Ekya NSDI 2022 paper:
```
@inproceedings {276952,
title = {Ekya: Continuous Learning of Video Analytics Models on Edge Compute Servers},
booktitle = {19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)},
year = {2022},
address = {Renton, WA},
url = {https://www.usenix.org/conference/nsdi22/presentation/bhardwaj},
publisher = {USENIX Association},
month = apr,
}
```
