Change ```OPT``` to ```True``` if simulation speed-up is needed.
Usage
```bash
# Simulate on Waymo dataset
cd {ekya project path}
python simulation --dataset waymo \
	--root /data/zxxia/ekya/results/waymo_profiles/20200619_1454 \
	--camera_names phx_0_9 phx_0_9  \
	--retraining_periods 100  \
    --hyp_map_path /data2/zxxia/ekya/ekya/experiment_drivers/profiling/hyp_map_all.json \
    --hyperparameters 0 1 2 3 4 5 \
    --num_tasks 10 \
	--output_path test_sim \
	--real_inference_profiles /data/zxxia/ekya/code/ekya/zxxia_dev/simulation/real_inference_profiles.csv

# Simulate on CityScapes dataset
cd {ekya project path}
python simulation --dataset cityscapes \
	--root /data2/zxxia/ekya/ekya/experiment_drivers/cityscapes_outputs/golden_profiles_all_hyper/ \
	--camera_names aachen bochum bremen cologne tubingen dusseldorf darmstadt jena monchengladbach zurich \
	--retraining_periods 200 \
    --hyp_map_path /data2/zxxia/ekya/ekya/experiment_drivers/profiling/hyp_map_all.json \
	--hyperparameters 0 1 2 3 4 5 \
	--output_path simulation_results/ \
	--real_inference_profiles /data/zxxia/ekya/code/ekya/zxxia_dev/simulation/real_inference_profiles.csv
```
