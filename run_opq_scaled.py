from vecs_io import *
from run_pq import execute
from opq import OPQ
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

    scale = np.max(np.linalg.norm(X, axis=1))
    X /= scale

    if T is None:
        T = X[:args.train_size]
    else:
        T = T[:args.train_size]
        T /= scale

    T = np.ascontiguousarray(T, np.float32)

    # pq, rq, or component of norm-pq
    quantizer = OPQ(M=args.num_codebook, Ks=args.Ks)
    if args.rank:
        execute(quantizer, X, T, Q, G, args.metric)
    if args.save_model:
        if not args.rank:
            quantizer.fit(T, iter=20)
        with open(args.save_dir + '/' + args.dataset + '_opq' + args.result_suffix + '.pickle', 'wb') as f:
            pickle.dump(quantizer, f)
