import json
from functools import wraps
from pathlib import Path
from typing import Callable, Tuple, List, Dict, Optional

from torchvision.datasets import ImageNet, DatasetFolder


def json_cache(filename: str):
    def decorator(f: Callable):
        @wraps(f)
        def wrapper(directory: str, *args, **kwargs):
            cache_path = Path(directory) / filename
            if cache_path.is_file():
                print(f'use cached {cache_path}')
                result = json.load(open(cache_path, 'r'))
            else:
                print(f'no cached {cache_path} available')
                result = f(directory, *args, **kwargs)
                json.dump(result, open(cache_path, 'w'))
            return result
        return wrapper
    return decorator


class CachedImageNet(ImageNet):
    @staticmethod
    @json_cache(filename='dataset_cache.json')
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        return DatasetFolder.make_dataset(directory, class_to_idx, extensions, is_valid_file)
