from vecs_io import *
from transformer import *
import math
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--data_dir', type=str, help='directory storing the data', default='./data/')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--data_type', type=str, default='fvecs', help='data type of base and queries')
    parser.add_argument('--topk', type=int, help='topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth, euclid by default')

    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')

    parser.add_argument('--chunk_size', type=int, help='chunk size', default=1000000)

    args = parser.parse_args()

    X, _, _, _ = mmap_loader(args.dataset, args.topk, args.metric, folder=args.data_dir, data_type=args.data_type)

    scale = np.max(np.linalg.norm(X, axis=1))
    X = X / scale

    np.random.seed(123)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_decoded', 'rb') as f:
        decoded = np.fromfile(f, dtype=np.float32).reshape(X.shape)

    residues = np.linalg.norm(X - decoded, axis=1)
    
    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_residues', 'wb') as f:
        residues.tofile(f)

