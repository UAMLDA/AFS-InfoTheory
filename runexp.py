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
import pandas as pd 
import skfeature as skf

import argparse

from utils import kuncheva, jaccard

# setup program constants 
data_folder = 'data/'

def main(args):

    # load normal and adversarial data 
    df_normal = pd.read_csv('data/clean/' + args.data + '.csv')
    df_adversarial = pd.read_csv('data/attacks/' + args.data + '_[xiao][' + str(.5) + '].csv') 

    # load adversarial data
    n = 10000
    n_train = 1000

    for k in range(args.cv): 
        # shuffle up the data for the experiment 
        k
    # average out stability

    # write the results to a file
    return None

if __name__ == '__main__': 
    # set up the parser
    parser = argparse.ArgumentParser(description='Run the experiments for the IJCNN 2021 paper.', 
                                     prog='runexp', 
                                     usage='%(prog)s [options]',) 
    parser.add_argument('-c', 
                        '--cv', 
                        type=int, 
                        default=5, 
                        help='cross-validation parameter [int]')
    parser.add_argument('-d', 
                        '--data', 
                        type=str, 
                        help='dataset name without csv or the path [str]')
    parser.add_argument('-p', 
                        '--poison', 
                        type=float, 
                        help='percentage of poison data to add [float]')
    parser.add_argument('-b', 
                        '--box', 
                        type=float, 
                        help='bounding box [float]. note the file must be in the data folder.')
    parser.add_argument('-o', 
                        '--output', 
                        type=str, 
                        default='results/',
                        help='output [str]')
    args = parser.parse_args()
    # run
    main(args)