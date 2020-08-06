import numpy as np
import json
import argparse
import os

import librosa

import soundfile as sf
class NumpyDataProvider:

    def __init__(self, filelist,
                 audio_dir, audio_ext, audio_dim,
                 min_samples=8000, max_samples=64000,
                 return_basename=False):


        self.audio_dir = audio_dir
        self.audio_ext = audio_ext
        self.audio_dim = audio_dim

        self.min_samples = min_samples
        self.max_samples = max_samples

        self.return_basename = return_basename

        
        if type(filelist) == list:
            # filelist is a python list
            self.filelist = filelist
        else:
            # filelist is list file
            with open(filelist) as f:
                self.filelist = f.readlines()
            self.filelist = [x.strip() for x in self.filelist]
        
        self.n_files = len(self.filelist)

    def __iter__(self):
        self._iter_ind = 0
        self._rand_ind = np.random.permutation(self.n_files)
        self._data_list = []
        return self

    def __next__(self):

        if len(self._data_list) == 0:
            n_samples = 0
            while n_samples < self.min_samples:
                
                if self._iter_ind >= self.n_files:
                    raise StopIteration
                i = self._rand_ind[self._iter_ind]
                bname = self.filelist[i]
                self._current_bname = bname
                audio_file = os.path.join(self.audio_dir, bname + self.audio_ext)
                
                try:
                    audio, fs = sf.read(audio_file)
                except RuntimeError as e:
                    if 'Error opening' in str(e):
                        print(e)
                        self._iter_ind += 1 
                        continue
                    else:
                        raise e

                # normalize
                audio /= np.max(np.abs(audio))
                # trim silences 
                audio, _ = librosa.effects.trim(
                    audio, top_db=30,
                    frame_length=512, hop_length=128
                )
                
                n_samples = audio.shape[0]
                self._iter_ind += 1 

            min_samples = self.min_samples
            max_samples = self.max_samples
            while audio.shape[0] >= min_samples:
                audio_part = audio[:max_samples]
                self._data_list.append(audio_part)
                audio = audio[max_samples:]

        if len(self._data_list) == 0:
            raise ValueError("Data list is still empty")

        audio = self._data_list.pop()

        if self.return_basename:
            return audio, self._current_bname
        else:
            return audio    

    def next(self):
        return self.__next__()       

class TestDataProvider:

    def __init__(self, filelist_path,
                 cond_dir, cond_ext, cond_dim,
                 file_format='raw'):

        self.filelist_path = filelist_path

        self.cond_dir = cond_dir
        self.cond_ext = cond_ext
        self.cond_dim = cond_dim

        self.file_format = file_format

        self.filelist = os.listdir(self.cond_dir)   
        self.filelist = [os.path.splitext(x)[0] for x in self.filelist]
        self.n_files = len(self.filelist)

    def __iter__(self):
        self._iter_ind = 0
        return self

    def __next__(self):
        if self._iter_ind >= self.n_files:
            raise StopIteration
        i = self._iter_ind
        bname = self.filelist[i]
        cond_file = os.path.join(self.cond_dir, bname + self.cond_ext)

        if self.file_format == 'raw':
            cond = np.fromfile(cond_file, dtype=np.float32).reshape([1, -1, self.cond_dim])
        elif self.file_format == 'npz':
            data = np.load(cond_file)
            cond = data['mel_spec']
            if cond.shape[1] != self.cond_dim:
                raise ValueError("Cond dimension mismatch" \
                    + f"    Expected {self.cond_dim}, read {cond.shape[1]}")   
        else:
            raise ValueError(f"Invalid cond file format '{self.file_format}'\n" \
                + "    valid options are 'raw' and 'npz' ")

        self._iter_ind += 1 

        # resample data
        cond_resamp_list = []
        orig_sr = int(1/0.0125) # tacotron frame rate
        target_sr = int(1/0.005) # model frame rate
        for col in cond.T:
            col_resamp = librosa.resample(col, orig_sr, target_sr)
            cond_resamp_list.append(np.expand_dims(col_resamp, axis=1))

        cond = np.concatenate(cond_resamp_list, axis=1)            

        return cond, bname            
