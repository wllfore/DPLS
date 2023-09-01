This is sample code of our paper [Dual-perspective Label Uncertainty Guidance for Accurate Tumor Segmentation] on the LiTS dataset.
Requirements
1) Environment: Python 3.9, Ubuntu 20.04, pytorch 1.8.0
2) pip install segmentation-models-pytorch
then overwrite with our modified version ‘code/segmentation_models_pytorch’
3) Other major dependencies: SimpleITK, cv2, medpy
Usage
1) The LiTS dataset folder is required to contain the following: volumes/ (volume files), labels/ 
(label mask files), train_vol.txt (volume ids for model training), valid_vol.txt (volume ids for 
model validation) and test_vol.txt (volume ids for model testing).
2) Run the command ‘code/luts_unet/run_luts.sh’, it has two steps: first, prepare training data for 
model input as well as DSL masks; second, run the model training script.
