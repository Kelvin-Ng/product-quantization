from aq import AQ
import argparse
import pickle
import numpy as np

if __name__ == '__main__':
    dataset = 'netflix'
    codebook = 4
    Ks = 256

    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name', default=dataset)
    parser.add_argument('--num_codebook', type=int, help='number of codebooks', default=codebook)
    parser.add_argument('--Ks', type=int, help='number of centroids in each quantizer', default=Ks)
    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')
    args = parser.parse_args()

    quantizer = AQ(M=args.num_codebook, Ks=args.Ks)

    with open(args.save_dir + '/' + args.dataset + '_aq' + args.result_suffix + '_codewords', 'rb') as f:
        quantizer.codewords = np.fromfile(f, dtype='float32').reshape(quantizer.M, quantizer.Ks, -1)

    with open(args.save_dir + '/' + args.dataset + '_aq' + args.result_suffix + '.pickle', 'wb') as f:
        pickle.dump(quantizer, f)
