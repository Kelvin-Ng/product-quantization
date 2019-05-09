import argparse
import pickle
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--save_dir', type=str, help='dir to save results', default='./results')
    parser.add_argument('--result_suffix', type=str, help='suffix to be added to the file names of the results', default='')

    args = parser.parse_args()

    np.random.seed(123)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '.pickle', 'rb') as f:
        quantizer = pickle.load(f)

    with open(args.save_dir + '/' + args.dataset + args.result_suffix + '_codewords', 'wb') as f:
        quantizer.codewords.tofile(f)


