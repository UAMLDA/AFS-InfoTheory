# AFS-InfoTheory

---

To run `scripts\poison_clean_data.py` you need to `pip install -r requirements.txt`, and also need to have [poisoning](https://github.com/rpgolota/poisoning/) installed from github, as it is not on PyPi.

Current attack creation progress is about 25%, but that is misleading since it is going from smallest to largest in terms of sample size.

- To create attacks, run the script `scripts\poison_clean_data.py`. It should resume where it left off.
- Currently it is running from the smallest to the largest datasets located in `data\filenames_by_samples.txt`.
- Attack file naming convention is. All output files go to `data\attacks`. All original files are in `data\clean`.
  - (filename no path)\_attack\_[(projection)].(extension)