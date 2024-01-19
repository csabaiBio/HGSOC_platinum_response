# generate random integer
rand_int=$((1 + RANDOM % 10000))

# exp name is `task_1_brca_subtype` and the randint_appended
exp_name="FFPE_level_0_BRCA_VS_WT__lr5_20_epoc_${rand_int}"

CUDA_VISIBLE_DEVICES=0,1,2 python CLAM/main.py --task task_1_tumor_vs_normal --split_dir splits/task_1_tumor_vs_normal_100\
 --exp_code $exp_name\
 --log_data\
 --subtyping\
 --bag_loss ce\
 --inst_loss svm\
 --model_type clam_sb\
 --max_epoch 20\
 --k 5\