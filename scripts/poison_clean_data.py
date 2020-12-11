import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import poisoning
import os

from rich.progress import Progress

DATA_PATH = '../data'
CLEAN_PATH = os.path.join(DATA_PATH, 'clean')
ATTACKS_PATH = os.path.join(DATA_PATH, 'attacks')
FILENAMES_PATH = 'filenames_by_samples.txt'

POISON_PERCENT = 0.2
POISON_PROJECTIONS = [0.5, 1, 1.5, 2, 2.5, 5]

VERBOSE = True

def write_attacks(filename, poisoned, labels):
    d1 = pd.DataFrame(poisoned)
    d2 = pd.DataFrame(labels)
    final = pd.concat([d1, d2], axis=1)
    final.to_csv(os.path.join(ATTACKS_PATH, filename), sep=',', header=None, index=None)

# for getting the filename of the attack file
def create_filename(filename, projection):
    file, extension = os.path.splitext(os.path.basename(filename))
    return file + '_attack_[' + str(projection) + ']' + extension

# function to poison a dataset, gets passed a function to update progress bar
def poison_clean(filename, progress_update):

    dataset = pd.read_csv(os.path.join(CLEAN_PATH, filename), sep=",", header=None)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,-1].values
    
    for projection in POISON_PROJECTIONS:
        new_filename = create_filename(filename, projection)
        
        if VERBOSE:
            print(f'    + Projection: {projection}')
        
        # if file exists already, skip
        if os.path.exists(os.path.join(ATTACKS_PATH, new_filename)):
            progress_update()
            continue
        
        model = poisoning.xiao2018()
        poisoned, labels = model.autorun(X, Y, POISON_PERCENT, projection)
        write_attacks(new_filename, poisoned, labels)
        progress_update()

# Main entry point for script
def main():
    with open(os.path.join(DATA_PATH, FILENAMES_PATH), 'r') as f:
        files = [file.strip() for file in f.readlines()]
        
    with Progress() as progress:
        t1 = progress.add_task("[bold red]Running Files...", total=len(files))
        t2 = progress.add_task("[bold blue]Running Models...", total=len(POISON_PROJECTIONS))
        def update():
            progress.advance(t2)
        for file in files:
            if VERBOSE:
                print(f'File: {file}')
            poison_clean(file, update)
            progress.advance(t1)
            progress.reset(t2)

if __name__ == "__main__":
    main()