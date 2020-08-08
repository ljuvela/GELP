# GAN-Excited Linear Prediction

Paper available at https://arxiv.org/abs/1904.03976, accepted for publication at Interspeech'19

Audio samples https://users.aalto.fi/~ljuvela/interspeech19/

I'm hoping to share the code before the Interspeech conference.

## Installing dependencies

conda env create -f environment-cpu.yml 



## Copy-synthesis with a pre-trained model
python generate_copysyn.py --model_dir sessions/42 --input_wav_dir demo_data/wav

## Adapting a pre-trained model

## Training from scratch

python make_filelist.py --input_wav_dir \ demo_data/wav --filelist_name demo_data/train_list_demo.txt

python train.py --config config/config.json 

### Note on TensorFlow versions

These scripts were written using TensorFlow 1.X and from TensorFlow 1.14 start to yield a number of deprecation warnings regarding TensorFlow 2.0 and above. Making the scripts compatible with TF 2 is not trivial due to major changes in static vs dynamic graphs and removal of global namespacing.

If you'd like to suppress the warnings, add the following lines after tensorflow imports in main.
```
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```