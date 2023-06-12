from argparse import ArgumentParser

from train import train
from tv.common import get_num_patches_reserved
from tv.config import get_config


def train_deit(keep_rate: float, temperature: float, data_root: str):
    config = get_config()
    model_name = 'deit_small_patch16_224'
    config['model']['args']['model_name'] = model_name
    config['data_transform']['model_name'] = model_name
    config['model']['args']['mask_patches_config']['num_patches_reserved'] = get_num_patches_reserved(keep_rate)
    config['model']['args']['mask_patches_config']['temperature'] = temperature
    config['train_dataset']['args']['root'] = data_root
    config['val_dataset']['args']['root'] = data_root
    config['test_dataset']['args']['root'] = data_root
    train(config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--keep_rate', type=float, required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    train_deit(args.keep_rate, args.temperature, args.data_root)
