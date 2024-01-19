# generate random integer
rand_int=$((1 + RANDOM % 10000))

# exp name is `task_1_brca_subtype` and the randint_appended
exp_name="HGSOC_SurvPath_hosp_split_try_Mayo_5_epocs"

CUDA_VISIBLE_DEVICES=0 python CLAM/main.py --task HGSOC_multimodal --split_dir splits/Mayo\
 --exp_code $exp_name\
 --log_data\
 --subtyping\
 --bag_loss ce\
 --inst_loss svm\
 --model_type MCAT_Surv\
 --max_epoch 5\
 --weighted_sample\
 --k 5\
 --omic_sizes [1,2,33,14,14,89]
