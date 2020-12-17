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

# -----------------------------------------------------------------------
# # setup program constants 
# percentage of poisoning levels  
POI_RNG = [.01, .025, .05, .075, .1, .125, .15, .175, .2]
# total number of poisoning levels 
NPR = len(POI_RNG)
# percentage of features that we want to select 
SEL_PERCENT = .1
# number of algorithms that we are going to test [JMI, MIM, MRMR, MIFS]
NALG = 4
# used when we select features 
FEAT_IDX = 0
# number of cross validation runs to perform
CV = 5
# dataset names 
DATA = [
        'conn-bench-sonar-mines-rocks',
        'ionosphere',
        'bank',
        'oocytes_trisopterus_nucleus_2f', 
        'statlog-german-credit', 
        'molec-biol-promoter', 
        'miniboone', 
        'ozone', 
        'spambase',
        'parkinsons', 
        'connect-4', 
        'oocytes_merluccius_nucleus_4d',
        'musk-1', 
        'musk-2', 
        'chess-krvkp', 
        'twonorm'
        ]
BOX = ['0.5', '1', '1.5', '2', '2.5', '5']
# -----------------------------------------------------------------------


def experiment(data, box, cv, output):
    """
    Write the results of an experiment.
        This function will run an experiment for a specific dataset for a bounding box. 
        There will be CV runs of randomized experiments run and the outputs will be 
        written to a file. 

        Parameters
        ----------
        data : string
            Dataset name.
            
        box : string 
            Bounding box on the file name.
        cv : int 
            Number of cross validation runs. 
            
        output : string
            If float or tuple, the projection will be the same for all features,
            otherwise if a list, the projection will be described feature by feature.
                    
        Returns
        -------
        None
            
        Raises
        ------
        ValueError
            If the percent poison exceeds the number of samples in the requested data.
    """
    #data, box, cv, output = 'conn-bench-sonar-mines-rocks', '1', 5, 'results/test.npz'

    # load normal and adversarial data 
    path_adversarial_data = 'data/attacks/' + data + '_[xiao][' + box + '].csv'
    df_normal = pd.read_csv('data/clean/' + data + '.csv', header=None).values
    df_adversarial = pd.read_csv(path_adversarial_data, header=None).values
    
    # separate out the normal and adversarial data 
    Xn, yn = df_normal[:,:-1], df_normal[:,-1]
    Xa, ya = df_adversarial[:,:-1], df_adversarial[:,-1]

    # change the labels from +/-1 to [0,1]
    ya[ya==-1], yn[yn==-1] = 0, 0

    # set up the data numers 
    p0, p1 = 1./cv, (1. - 1./cv)
    N = len(Xn)
    Ntr, Nte = int(p1*N), int(p0*N)
    n_selected_features = int(Xn.shape[1]*SEL_PERCENT)+2

    err_jaccard, err_kuncheva = np.zeros((NPR, NALG)), np.zeros((NPR, NALG))    

    for _ in range(cv): 
        # shuffle up the data for the experiment 
        i = np.random.permutation(N)
        Xtrk, ytrk, Xtek, ytek = Xn[i][:Ntr], yn[i][:Ntr], Xn[i][Nte:], yn[i][Nte:]

        # run feature selection on the baseline dataset without an adversarial data 
        sf_base_jmi, sf_base_mim, sf_base_mrmr, sf_base_mifs = \
            JMI.jmi(Xtrk, ytrk, n_selected_features=n_selected_features)[FEAT_IDX], \
            MIM.mim(Xtrk, ytrk, n_selected_features=n_selected_features)[FEAT_IDX], \
            MRMR.mrmr(Xtrk, ytrk, n_selected_features=n_selected_features)[FEAT_IDX], \
            MIFS.mifs(Xtrk, ytrk, n_selected_features=n_selected_features)[FEAT_IDX]

        for n in range(NPR): 
            Np = int(len(ytrk)*POI_RNG[n]+1)
            if Np >= len(ya): 
                ValueError('Number of poison data requested is larger than the available data.')
            Nn = len(ytrk) - Np
            idx_normal, idx_adversarial = np.random.permutation(len(ytrk))[:Nn], \
                                          np.random.permutation(len(ya))[:Np]
            Xtrk_poisoned, ytrk_poisoned = np.concatenate((Xtrk[idx_normal], Xa[idx_adversarial])), \
                                           np.concatenate((ytrk[idx_normal], ya[idx_adversarial]))
            sf_adv_jmi, sf_adv_mim, sf_adv_mrmr, sf_adv_mifs = \
                JMI.jmi(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX], \
                MIM.mim(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX], \
                MRMR.mrmr(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX], \
                MIFS.mifs(Xtrk_poisoned, ytrk_poisoned, n_selected_features=n_selected_features)[FEAT_IDX]

            # calculate the accumulated jaccard and kuncheva performances for each of the 
            # feature selection algorithms 
            err_jaccard[n, 0] += jaccard(sf_adv_mim, sf_base_mim)
            err_jaccard[n, 1] += jaccard(sf_adv_mifs, sf_base_mifs)
            err_jaccard[n, 2] += jaccard(sf_adv_mrmr, sf_base_mrmr)
            err_jaccard[n, 3] += jaccard(sf_adv_jmi, sf_base_jmi)

            err_kuncheva[n, 0] += kuncheva(sf_adv_mim, sf_base_mim, Xtrk.shape[1])
            err_kuncheva[n, 1] += kuncheva(sf_adv_mifs, sf_base_mifs, Xtrk.shape[1])
            err_kuncheva[n, 2] += kuncheva(sf_adv_mrmr, sf_base_mrmr, Xtrk.shape[1])
            err_kuncheva[n, 3] += kuncheva(sf_adv_jmi, sf_base_jmi, Xtrk.shape[1])

    err_jaccard,  err_kuncheva = err_jaccard/cv, err_kuncheva/cv
    np.savez(output, err_jaccard=err_jaccard, err_kuncheva=err_kuncheva)

    return None

if __name__ == '__main__': 

    for data in DATA: 
        for box in BOX: 
            print('Running ' + data)
            try: 
                experiment(data, box, CV, 'results/' + data + '_[xiao][' + box + ']_results.npz')
            except: 
                print(' ... ERROR ...')
