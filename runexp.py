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
from skfeature.function.information_theoretical_based import JMI, MIM, MRMR, MIFS

# setup program constants 
data_folder = 'data/'
POI_RNG = [.01, .025, .05, .075, .1, .125, .15, .175, .2]
NPR = len(POI_RNG)
SEL_PERCENT = .1
NALG = 4
FEAT_IDX = 0

def main():
    #data, box, cv = args.data, args.box, args.cv
    data, box, cv = 'blood', '1', 5

    # load normal and adversarial data 
    path_adversarial_data = 'data/attacks/' + data + '_[xiao][' + box + '].csv'
    df_normal = pd.read_csv('data/clean/' + data + '.csv', header=None).values
    df_adversarial = pd.read_csv(path_adversarial_data, header=None).values
    Xn, yn = df_normal[:,:-1], df_normal[:,-1]
    Xa, ya = df_adversarial[:,:-1], df_adversarial[:,-1]

    # change the labels 
    ya[ya==-1], yn[yn==-1] = 0, 0

    # set up the data numers 
    p0, p1 = 1./cv, (1. - 1./cv)
    N = len(Xn)
    Ntr, Nte = int(p1*N), int(p0*N)
    n_selected_features = int(Xn.shape[1]*SEL_PERCENT)+2
    Npoi = np.array(POI_RNG)*Ntr

    idx_jaccard, idx_kuncheva = np.zeros((NPR, NALG)), np.zeros((NPR, NALG))    

    for k in range(cv): 
        # shuffle up the data for the experiment 
        i = np.random.permutation(N)
        Xtrk, ytrk, Xtek, ytek = Xn[i][:Ntr], yn[i][:Ntr], Xn[i][Nte:], yn[i][Nte:]

        # run feature selection on the baseline dataset without an adversarial data 
        sf_base_jmi, sf_base_mim, sf_base_mrmr, sf_base_mifs = \
            JMI.jmi(Xtrk, ytrk, n_selected_features=n_selected_features), \
            MIM.mim(Xtrk, ytrk, n_selected_features=n_selected_features), \
            MRMR.mrmr(Xtrk, ytrk, n_selected_features=n_selected_features), \
            MIFS.mifs(Xtrk, ytrk, n_selected_features=n_selected_features)

        print(jaccard(sf_base_jmi[FEAT_IDX], sf_base_mim[FEAT_IDX]))
    # average out stability

    # write the results to a file
    return None

if __name__ == '__main__': 
    # set up the parser
    '''
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
                        type=str, 
                        help='bounding box [str]. note the file must be in the data folder.')
    parser.add_argument('-o', 
                        '--output', 
                        type=str, 
                        default='results/',
                        help='output [str]')
    args = parser.parse_args()
    # run
    main(args)
    '''
    main()