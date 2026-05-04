"""
40×40 Hi-C super-resolution prediction (HiCARN-style).

Resolves ``Models``, ``Utils``, and ``Arg_Parser`` when run from the repository
root without manual PYTHONPATH setup. Expects the same CLI as the original
``40x40_Predict.py`` in HiCARN (see ``data/Arg_Parser.data_predict_parser``).
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
for _p in (_REPO, _REPO / "data"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
import Models.DeepHiC as DeepHiC
import Models.HiCARN_1 as HiCARN_1
import Models.HiCARN_2 as HiCARN_2
from Utils.io import spreadM, together
from Arg_Parser import *


def dataloader(data, batch_size=64):
    inputs = torch.tensor(data['data'], dtype=torch.float)
    target = torch.tensor(data['target'], dtype=torch.float)
    inds = torch.tensor(data['inds'], dtype=torch.long)
    dataset = TensorDataset(inputs, target, inds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def data_info(data):
    indices = data['inds']
    compacts = data['compacts'][()]
    sizes = data['sizes'][()]
    return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
    info_str = filename.split('.')[0].split('_')[2:-1]
    chunk = get_digit(info_str[0])
    stride = get_digit(info_str[1])
    bound = get_digit(info_str[2])
    scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
    return chunk, stride, bound, scale


def hicarn_predictor(model, hicarn_loader, ckpt_file, device):
    print(model)
    deepmodel = model.Generator(num_channels=64).to(device)
    if not os.path.isfile(ckpt_file):
        ckpt_file = f'save/{ckpt_file}'
    deepmodel.load_state_dict(torch.load(ckpt_file, map_location=torch.device('cpu')))
    print(f'Loading HiCARN checkpoint file from "{ckpt_file}"')

    result_data = []
    result_inds = []

    deepmodel.eval()
    with torch.no_grad():
        for batch in tqdm(hicarn_loader, desc='HiCARN Predicting: '):
            lr, hr, inds = batch
            lr = lr.to(device)
            out = deepmodel(lr)

            result_data.append(out.to('cpu').numpy())
            result_inds.append(inds.numpy())
    result_data = np.concatenate(result_data, axis=0)
    result_inds = np.concatenate(result_inds, axis=0)
    hicarn_hics = together(result_data, result_inds, tag='Reconstructing: ')
    return hicarn_hics


def save_data(carn, compact, size, file):
    carn = spreadM(carn, compact, size, convert_int=False, verbose=True)
    np.savez_compressed(file, hicarn=carn, compact=compact)
    print('Saving file:', file)


if __name__ == '__main__':
    args = data_predict_parser().parse_args(sys.argv[1:])
    cell_line = args.cell_line
    low_res = args.low_res
    ckpt_file = args.checkpoint
    cuda = args.cuda
    model = args.model
    HiCARN_file = args.file_name
    print('WARNING: Predict process requires large memory, thus ensure that your machine has ~150G memory.')
    if multiprocessing.cpu_count() > 23:
        pool_num = 23
    else:
        exit()

    # Prefer data_40_40 layout used by this repo's data/Generate.py; fall back to legacy ``data``.
    in_dir = None
    for sub in ("data_40_40", "data"):
        cand = os.path.join(root_dir, cell_line, sub)
        if os.path.isfile(os.path.join(cand, HiCARN_file)):
            in_dir = cand
            break
    if in_dir is None:
        legacy = os.path.join(root_dir, "data")
        if os.path.isfile(os.path.join(legacy, HiCARN_file)):
            in_dir = legacy
        else:
            raise FileNotFoundError(
                f"Could not find {HiCARN_file!r} under {root_dir!r} "
                f"(tried {cell_line}/data_40_40, {cell_line}/data, data)."
            )

    out_dir = os.path.join(root_dir, 'predict', cell_line)
    mkdir(out_dir)

    chunk, stride, bound, scale = filename_parser(HiCARN_file)

    device = torch.device(
        f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
    print(f'Using device: {device}')

    start = time.time()
    print(f'Loading data[HiCARN]: {HiCARN_file} from {in_dir}')
    hicarn_data = np.load(os.path.join(in_dir, HiCARN_file), allow_pickle=True)
    hicarn_loader = dataloader(hicarn_data)

    indices, compacts, sizes = data_info(hicarn_data)

    if model == "HiCARN_1":
        model = HiCARN_1

    if model == "HiCARN_2":
        model = HiCARN_2

    if model == "DeepHiC":
        model = DeepHiC

    hicarn_hics = hicarn_predictor(model, hicarn_loader, ckpt_file, device)

    def save_data_n(key):
        file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
        save_data(hicarn_hics[key], compacts[key], sizes[key], file)

    pool = multiprocessing.Pool(processes=pool_num)
    print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
    for key in compacts.keys():
        pool.apply_async(save_data_n, (key,))
    pool.close()
    pool.join()
    print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
