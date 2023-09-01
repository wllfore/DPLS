### 1. data prepare
python lits_data_process.py --data_dir ~/data/LiTS --run_mode 1

### 2. model training 
CUDA_VISIBLE_DEVICES=0 python train_lits.py --tumor_dir ~/data/LiTS/tumor --encoder mit_b3 --max_epochs 40 --job_name luts_mitb3 --dsl_mode 1 --msl_mode 1 --atsw_mode 1 --loss_mode 4