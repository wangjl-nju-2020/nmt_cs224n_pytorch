# @Author: wangjl
# @Email: If you need any help, feel free to contact me via wangjl.nju.2020@gmail.com.

from argparse import Namespace


DATA_ROOT = '/home/wangjl/MT/en_es_data/'
WORK_DIR = '/home/wangjl/MT/work_dir/'


args = {
    # train parameter
    'fixed_seed': 0,
    'batch_size': 32,
    'num_workers': 5,
    'device': 'cuda: 0',
    'embed_size': 512,
    'hidden_size': 512,
    'dropout_rate': 0.3,
    'lr': 4e-5,
    'lr_decay': 0.5,
    'max_epochs': 200,
    'clip_gradient': 5.0,
    'valid_niter': 10,
    'batch_val_size': 32,
    'patience': 5,
    'max_num_trial': 5,
    'max_decoding_time_step': 50,
    'beam_size': 5,
    'threshold': 5,

    # data_dir
    'train_src_path': DATA_ROOT + 'train.en',
    'train_dst_path': DATA_ROOT + 'train.es',
    'val_src_path': DATA_ROOT + 'dev.en',
    'val_dst_path': DATA_ROOT + 'dev.es',
    'test_src_path': DATA_ROOT + 'test.en',
    'test_dst_path': DATA_ROOT + 'test.es',

    # work_dir
    'vocab_root': WORK_DIR,
    'train_src_pkl': WORK_DIR + 'train.en.pkl',
    'train_dst_pkl': WORK_DIR + 'train.es.pkl',
    'test_res_path': WORK_DIR + 'test.res',
    'model_save_path': WORK_DIR + 'best_model.optim',
    'optimizer_save_path': WORK_DIR + 'best_model_optimizer.optim'
}

hparams = Namespace(**args)
