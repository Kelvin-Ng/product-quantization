from vecs_io import *
from buckets_io import *
import pickle
import numpy as np
import argparse

def bucket(M, Ks, vecs, codewords, codes):
    if vecs is not None:
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
    if codewords is not None:
        assert codewords.shape == (M, Ks, D)

    N = len(codes)

    buckets_items = [[] for i in range(Ks ** M)]
    for i in range(N):
        code = 0
        for m in reversed(range(M)):
            code *= Ks
            code += codes[i, m]

        buckets_items[code].append(i)

    if vecs is not None and codewords is not None:
        buckets_residue = np.zeros((N), dtype=vecs.dtype)

        for code in range(Ks ** M):
            code_tmp = code
            center = np.zeros((D), dtype=vecs.dtype)
            for m in range(M):
                center += codewords[m][code_tmp % Ks]
                code_tmp //= Ks
            if len(buckets_items[code]) > 0:
                buckets_residue[code] = np.max(np.linalg.norm(vecs[buckets_items[code]] - center, axis=1))
            else:
                buckets_residue[code] = 0
            #buckets_residue[code] = 0
            #for item_id in buckets_items[code]:
            #    buckets_residue[code] = max(buckets_residue[code], np.linalg.norm(vecs[item_id] - center))

        return buckets_items, buckets_residue
    else:
        return buckets_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--data_dir', type=str, help='directory storing the data', default='./data/')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--data_type', type=str, default='fvecs', help='data type of base and queries')
    parser.add_argument('--topk', type=int, help='topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth, euclid by default')

    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')

    args = parser.parse_args()

    X, _, _, _ = mmap_loader(args.dataset, args.topk, args.metric, folder=args.data_dir, data_type=args.data_type)

    np.random.seed(123)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '.pickle', 'rb') as f:
        quantizer = pickle.load(f)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_encoded', 'rb') as f:
        codes = np.fromfile(f, dtype=quantizer.code_dtype).reshape(-1, quantizer.num_codebooks)

    rows, row_residues = bucket(1, quantizer.Ks, X, quantizer.codewords, codes)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_row_residues', 'wb') as f:
        row_residues.tofile(f)
