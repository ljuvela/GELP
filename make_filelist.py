import os
import argparse
from glob import glob
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()                   
    parser.add_argument('--input_wav_dir', type=str, required=True) 
    parser.add_argument('--filelist_name', type=str, required=True) 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    files = glob(os.path.join(args.input_wav_dir, '*.wav'))
    basenames = [os.path.splitext(os.path.basename(f))[0] for f in files]
    with open(args.filelist_name, 'w') as f:
        for bname in basenames:
            print(bname, file=f)
    