# AFS-InfoTheory


## Preparing the Data 

To run `scripts/poison_clean_data.py` you need to `pip install -r scripts/requirements.txt`.

- To create attacks, run the script `scripts/poison_clean_data.py`. It should resume where it left off.
- Currently it is running from the smallest to the largest datasets located in `data/filenames_by_samples.txt`.
- Attack file naming convention is:
  - (filename no path)\_[(type)][(projection)].(extension)
- All output files go to `data/attacks`. All original files are in `data/clean`.



## Running the Experiments  


To run the experiments, you need to install the same requirements above as well as [skfeature](https://jundongl.github.io/scikit-feature/). 
```
python runexp.py 
``` 