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