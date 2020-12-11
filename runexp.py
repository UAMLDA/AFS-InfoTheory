#!/usr/bin python 

# Copyright 2021 Gregory Ditzler 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.


import numpy as np 
import skfeature as skf

import argparse

from utils import kuncheva, jaccard


def main(args):

    # load normal data 
    # 
    # load adversarial data
    # 
    n = 10000
    n_train = 1000

    for k in range(args.cv): 
        # shuffle up the data for the experiment 
        idx = np.random.permutation(n)
        idx_tr, idx_te = idx[:n_train], idx[n_train:]

        # sample poison percentage w/ args.poison

        # run baseline: JMI, MIM, mRMR, etc. 

        # measure stability (kuncheva, jaccard, noriega)

        # classification: acc, f1, etc [kNN, CART,...]

    # average out stability

    # write the results to a file
    return None

if __name__ == '__main__': 
    # set up the parser
    parser = argparse.ArgumentParser(description='Run the experiments for the IJCNN 2021 paper.') 
    parser.add_argument('cv', 
                        metavar='c', 
                        type=int, 
                        default=5)
    parser.add_argument('data', 
                        metavar='d', 
                        type=str)
    parser.add_argument('poison', 
                        metavar='p', 
                        type=float)
    parser.add_argument('output', 
                        metavar='o', 
                        type=str, 
                        default='results/')
    args = parser.parse_args()
    # run
    main(args)