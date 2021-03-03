# Branch: Extra


## Extra Changes:
- Max poisoning ratio: 0.3
- POI_RNG: [.01, .025, .05, .075, .1, .125, .15, .175, .2, .25, .3]
- BOX = ['0.5', '1', '1.5', '2', '2.5', '5', '10', '15', '20']
- CV = 10
- ALGS = [JMI, MIM, MRMR, MIFS, CMIM, DISR, ICAP]



## Bug
  - X_tr, y_tr, X_te, y_te = X[i][:Ntr], y[i][Ntr], X[i][Nte:], y[i][Nte:]
                    
  - X_tr, y_tr, X_te, y_te = X[i][:Ntr], y[i][:Ntr], X[i][-Nte:], y[i][-Nte:]      

## New files are:
  - runexp_extra.py
  - Extra_experiments contains plots and npz files
  - Extra_exp_data: contains clean data and attack data
  - Extra_plots.ipynb: notebook that contain code to generate plots
