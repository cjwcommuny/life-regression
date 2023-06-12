import os
from pathlib import Path

from tv.common import generate_id

os.chdir(Path(__file__).parent.parent.parent)
#
def get_config():
    id = generate_id()
    gpus = 1
    seed = 92479 # a random prime number
    fast_dev_run = False
    train_batch_size = 128
    train_rate = 1.0
    val_batch_size = 1
    test_batch_size = 32
    DATASET_TRAIN_SIZE = 1281167
    max_epochs = 1
    num_training_steps = int(train_rate * DATASET_TRAIN_SIZE / train_batch_size * max_epochs)
    model_name = None

    config = {
        'id': id,
        'seed': seed,
        'model': {
            'name': 'TimmVitAdapter',
            'args': {
                'model_name': model_name,
                'num_patches': 197,
                'pretrained': True,
                'freeze_vit': True,
                'unfreeze_vit_blocks_idxes': (),
                'mask_patches_config': {
                    'layer_indices': (4, 7, 10),
                    'hard_mask_training': False,
                    'pruning_eval': True,
                    'num_patches_reserved': (140, 100, 72),
                    'temperature': 1.0,
                },
                'base_lr': 1e-5,
                'extension_lr': 1e-3,
            }
        },
        'loss': {
            'label_smoothing': 0.1
        },
        'data_transform': {
            'model_name': model_name
        },
        'train_dataset': {
            'name': 'CachedImageNet',
            'args': {
                'root': None,
                'split': 'train'
            }
        },
        'val_dataset': {
            'name': 'CachedImageNet',
            'args': {
                'root': None,
                'split': 'val'
            }
        },
        'test_dataset': {
            'name': 'CachedImageNet',
            'args': {
                'root': None,
                'split': 'val'
            }
        },
        'train_dataloader': {
            'batch_size': train_batch_size,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
        },
        'val_dataloader': {
            'batch_size': val_batch_size,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
        },
        'test_dataloader': {
            'batch_size': test_batch_size,
            'shuffle': False,
            'num_workers': 4,
            'pin_memory': True,
        },
        'optimizer': {
            'type': 'AdamW',
            'args': {
                'lr': 1e-3,
                'weight_decay': 0.05
            }
        },
        'scheduler': {
            'type': 'Linear',
            'args': {
                'num_warmup_steps': num_training_steps / 100,
                'num_training_steps': num_training_steps
            },
            'interval': 'step'
        },
        'trainer': {
            'gpus': gpus,
            'track_grad_norm': -1,
            # 'accumulate_grad_batches': accumulate_grad_batches,
            'fast_dev_run': fast_dev_run,
            # 'amp_backend': 'native',
            # 'precision': 16,
            'benchmark': True,
            'deterministic': False,
            'check_val_every_n_epoch': 1,
            'gradient_clip_val': 0,
            # 'progress_bar_refresh_rate': 0,
            # 'val_check_interval': 1.0,
            'weights_summary': None,
            'terminate_on_nan': True,
            'log_every_n_steps': 50,
            #
            'max_epochs': max_epochs,
            'limit_train_batches': train_rate,
            'limit_val_batches': 0.0,
            'limit_test_batches': 1.0,
            # 'num_sanity_val_steps': 0
        },
        'logger': {
            'version': id,
            'save_dir': './log/vit-prune-patches'
        },
        'checkpoint_callback': {
            'save_last': True,
            'save_top_k': 1,
            'mode': 'max',
            'monitor': 'epochs/val-acc',
            'every_n_train_steps': 500
        },
    }
    return config
