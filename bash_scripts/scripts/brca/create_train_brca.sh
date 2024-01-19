# generate random integer
rand_int=$((1 + RANDOM % 10000))

# exp name is `task_1_brca_subtype` and the randint_appended
exp_name="task_1_brca_subtype_${rand_int}"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task task_1_brca_subtype --split_dir splits/task_1_brca_subtype_100\
 --exp_code $exp_name\
 --log_data\
 --subtyping\
 --bag_loss ce\
 --inst_loss svm\
 --model_type clam_sb\
 --max_epoch 50\