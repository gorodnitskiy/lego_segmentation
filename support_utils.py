from typing import Optional
import os
import random
import yaml

import numpy as np
import torch


def yaml_parser(
        path: Optional[str] = None,
        data: Optional[str] = None,
        loader: yaml.SafeLoader = yaml.SafeLoader
) -> dict:
    if path:
        with open(r"{}".format(path)) as file:
            return yaml.load(file, Loader=loader)

    elif data:
        return yaml.load(data, Loader=loader)

    else:
        raise ValueError('Either a path or data should be defined as input')


def freeze_seed(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.cuda.deterministic = deterministic
    torch.cuda.benchmark = benchmark


def set_max_threads(max_threads: int = 32) -> None:
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
