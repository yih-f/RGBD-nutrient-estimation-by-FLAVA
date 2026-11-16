# RGBD-nutrient-estimation-by-FLAVA
Paper: RGB-D food nutrient estimation supported by FLAVA contrastive learning

Dataset: Nutrition 5K

Model architecture
![architecture](https://github.com/user-attachments/assets/f42adb24-5a49-4f47-be3f-862d9d53097d)
Performance: mean PMAE 14.43%

## Repository layout

- `models/`: four Python training scripts for the baseline and FLAVA ablations.
- `notebooks/`: sample evaluation with Grad-CAM and depth-overlay exploration.
- `data/`: the tracked Nutrition5K metadata and train/test split lists. The RGB-D images themselves are not included here.
- `results/`: saved training-history CSV files and ablation run logs.
- `outputs/`: place for future generated artifacts; currently empty.

The training scripts still contain absolute paths for the original training environment near the top of each file. Set `rgbd_dir`, `metadata_file`, `train_ids_file`, and `test_ids_file` to your own locations before running them. Moving the tracked metadata into `data/` does not automatically update those settings.
