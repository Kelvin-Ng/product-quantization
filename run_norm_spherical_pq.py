from vecs_io import *
from pq_spherical import SphericalPQ
from pq_norm import NormPQ
from sorter import *
from run_pq import execute
from run_pq import parse_args
import pickle


if __name__ == '__main__':
    dataset = 'netflix'
    topk = 20
    codebook = 4
    Ks = 256
    metric = 'product'

    # override default parameters with command line parameters
    import sys
    args = parse_args(dataset, topk, codebook, Ks, metric)
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, Ks = {}, metric = {}"
          .format(args.dataset, args.topk, args.num_codebook, args.Ks, args.metric))

    X, T, Q, G = loader(args.dataset, args.topk, args.metric, folder=args.data_dir)
    if T is None:
        T = X[:args.train_size]
    else:
        T = T[:args.train_size]
    T = np.ascontiguousarray(T, np.float32)

    quantizer = SphericalPQ(M=args.num_codebook-1, Ks=args.Ks)
    quantizer = NormPQ(n_percentile=args.Ks, quantize=quantizer, recover='spherical')
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        if not args.rank:
            quantizer.fit(T, iter=20)
        with open(args.save_dir + '/' + args.dataset + '_norm_spherical_pq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)
