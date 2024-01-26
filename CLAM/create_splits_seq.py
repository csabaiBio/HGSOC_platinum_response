import os
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

def setup_task(args):
    if args.task == 'HGSOC_platinum_response_prediction':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = 'HGSOC_platinum_responce/HGSOC_TCGA_main.csv',    
                    data_dir=args.embeddings_path,    
                    omics_structure=args.omics_structure,
                    embed_dim=args.embed_dim,
                    shuffle = False, 
                    seed = args.seed, 
                    print_info = True,
                    label_dict = {'0': 0, '1': 1},
                    ignore=[])
        

    elif args.task == 'HGSOC_train_TCGA_test':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = '/mnt/ncshare/ozkilim/BRCA/data/tasks/HGSOC_TCGA_merged_all.csv',    
                    data_dir=args.embeddings_path,    
                    omics_structure=args.omics_structure,
                    embed_dim = args.embed_dim,
                    shuffle = False, 
                    seed = args.seed, 
                    print_info = True,
                    label_dict = {'0': 0, '1': 1},
                    ignore=[])
        

    elif args.task == 'NERO_HGSOC_test':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path =  "/mnt/ncshare/ozkilim/BRCA/data/tasks/nero_brca_hgsoc_test.csv",    
                    data_dir=args.embeddings_path,    
                    omics_structure=args.omics_structure,
                    embed_dim = args.embed_dim,
                    shuffle = False, 
                    seed = args.seed, 
                    print_info = True,
                    label_dict = {'0': 0, '1': 1},
                    ignore=[])

        
    elif args.task == 'TCGA_proteomic_Ov':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = '/mnt/ncshare/ozkilim/BRCA/data/tasks/TCGA_platinum_reposnce_test_set.csv',    
                    data_dir=args.embeddings_path, 
                    omics_structure=args.omics_structure,   
                    shuffle = False, 
                    seed = args.seed, 
                    print_info = True,
                    label_dict = {'0': 0, '1': 1},
                    ignore=[])
        

    elif args.task == 'COAD_PFS_580_FFPE':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = '/mnt/ncshare/ozkilim/BRCA/data/tasks/COAD_PFS_FFPE.csv',    
                    data_dir='/tank/WSI_data/TCGA_WSI_organized/TCGA-COAD/CLAM/level_0/ViT/',    
                    shuffle=False,
                    seed=args.seed,
                    print_info=True,
                    label_dict={'0': 0, '1': 1},
                    patient_strat=True,
                    label_col='PFS_binary',
                    )

    elif args.task == 'FFPE_BRCA':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = './data/tasks/BRCA_pos_neg_FFPE.csv',    
                    data_dir='/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT_stain_norm_Mancheko/',    
                    shuffle=False,
                    seed=args.seed,
                    print_info=True,
                    label_dict={'negative': 0, 'positive': 1},
                    patient_strat=True,
                    label_col='brca_status',
                    )
        
    elif args.task == 'BRCA_vs_quiescent':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = './data/tasks/BRCA_vs_quiescent_DX_TS.csv',    
                    data_dir='/tank/WSI_data/Ovarian_WSIs/TCGA-OV/CLAM/level_0/Ov_ViT_stain_norm_Mancheko/',    
                    shuffle=False,
                    seed=args.seed,
                    print_info=True,
                    label_dict={'negative': 0, 'positive': 1},
                    patient_strat=True,
                    label_col='brca_status',
                    )
        
    
    elif args.task == 'FFPE_BRCA_zoltan_test':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = '/mnt/ncshare/ozkilim/BRCA/data/tasks/ZS_test_list_prepared.csv',#list of slides...    
                    data_dir='/tank/WSI_data/Ovarian_WSIs/BRCA_blind_test/CLAM/level_0/Ov_ViT_stain_norm_Mancheko/',    
                    shuffle=False,
                    seed=args.seed,
                    print_info=True,
                    label_dict={'0': 0, '0': 1},
                    patient_strat=True,
                    label_col='label',
                    )


    elif args.task == 'HRD_status':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = './data/raw_subsets/HRD_binary.csv',    
                    data_dir='./data/embeds/KimiaNet_features',    
                    shuffle=False,
                    seed=args.seed,
                    print_info=True,
                    label_dict={'0': 0, '1': 1},
                    patient_strat=True,
                    label_col='hrd_status',
                    )


    elif args.task == 'pathomic_fusion_plat_responce_DSS':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = './data/tasks/combined_genomic_plat_responce.csv',    
                    data_dir='./data/embeds/KimiaNet_features',    
                    shuffle=False,
                    seed=args.seed,
                    print_info=True,
                    label_dict={'0.0': 0, '1.0': 1},
                    patient_strat=True,
                    label_col='DSS',
                    )
        
    

    elif args.task == 'platinum_DSS_survival':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(
                    csv_path = './data/raw_subsets/DSS_platinum_survival.csv',    
                    data_dir='./data/embeds/KimiaNet_features',    
                    shuffle=False,
                    seed=args.seed,
                    print_info=True,
                    label_dict={'0.0': 0, '1.0': 1},
                    patient_strat=True,
                    label_col='DSS',
                    )
                
    elif args.task == 'task_02_tumor_subtyping':
        args.n_classes=3
        dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                                patient_strat= True,
                                patient_voting='maj',
                                ignore=[])

    elif args.task == 'task_1_brca_subtype_FFPE':
        args.n_classes=3
        dataset = Generic_MIL_Dataset(
            csv_path = './data/raw_subsets/brca_dataset_FFPE.csv',
            data_dir='./data/embeds/KimiaNet_features/', 
            shuffle=True,
            seed=args.seed,
            print_info=True,
            label_dict={'intact': 0, 'negative': 1, 'positive': 2},
            patient_strat=False,
            label_col='brca_status',
            ignore=['hrd_score', 'slide_type'])
        
        dataset.load_from_h5(True)
        
    elif args.task == 'task_2_slide_type':
        args.n_classes=6
        #'BS1': 663, 'TS1': 652, 'DX1': 106, 'BS2': 54, 'TSA': 5, 'DX2': 1
        dataset = Generic_MIL_Dataset(
            csv_path = './data/brca_dataset.csv',
            data_dir='./data/embeds/KimiaNet_features_level_1/',
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'BS1': 0, 'TS1': 1, 'DX1': 2, 'BS2': 3, 'TSA': 4, 'DX2': 5},
            patient_strat=True,
            patient_voting='max',
            label_col='slide_type',
            ignore=['hrd_score', 'brca_status'])
        
        dataset.load_from_h5(True)
        
    elif args.task == 'task_1_hunCRC_resnet_categorical':
        args.n_classes=4
        dataset = Generic_MIL_Dataset(
            csv_path = './data/hunCRC_CLAM/categorical.csv',
            data_dir='./data/hunCRC_CLAM/CLAM_ResNet50_features/',
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'adenoma': 0, 'CRC': 1, 'non_neoplastic_lesion': 2, 'negative': 3},
            patient_strat=True,
            patient_voting='max',
            label_col='category',
            ignore=['slide_type'])
        
        dataset.load_from_h5(True)
        
    elif args.task == 'task_1_hunCRC_vit_categorical':
        args.n_classes=4
        dataset = Generic_MIL_Dataset(
            csv_path = './data/hunCRC_CLAM/categorical.csv',
            data_dir='./data/hunCRC_CLAM/CLAM_ViT_features/',
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'adenoma': 0, 'CRC': 1, 'non_neoplastic_lesion': 2, 'negative': 3},
            patient_strat=True,
            patient_voting='max',
            label_col='category',
            ignore=['slide_type'])
        
        dataset.load_from_h5(True)
        
    elif args.task == 'task_1_hunCRC_kimianet_categorical':
        args.n_classes=4
        dataset = Generic_MIL_Dataset(
            csv_path = './data/hunCRC_CLAM/categorical.csv',
            data_dir='./data/hunCRC_CLAM/CLAM_kimianet_features/',
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict={'adenoma': 0, 'CRC': 1, 'non_neoplastic_lesion': 2, 'negative': 3},
            patient_strat=True,
            patient_voting='max',
            label_col='category',
            ignore=['slide_type'])
        
        dataset.load_from_h5(True)

    else:
        raise NotImplementedError
    
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])

    output = {
        'dataset': dataset,
        'num_slides_cls': num_slides_cls,
        'args': args
    }

    try:
        val_num = np.round(num_slides_cls * args.val_frac).astype(int)
        test_num = np.round(num_slides_cls * args.test_frac).astype(int)

        output.update({
            'val_num': val_num,
            'test_num': test_num,
        })
    except AttributeError:
        pass

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
    parser.add_argument('--label_frac', type=float, default= 1.0,
                        help='fraction of labels (default: 1)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--k', type=int, default=10,
                        help='number of splits (default: 10)')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--val_frac', type=float, default= 0.1,
                        help='fraction of labels for validation (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default= 0.1,
                        help='fraction of labels for test (default: 0.1)')

    args = parser.parse_args()

    setup = setup_task(args)
    
    dataset = setup['dataset']
    val_num = setup['val_num']
    test_num = setup['test_num']
    num_slides_cls = setup['num_slides_cls']
    args = setup['args']

    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



