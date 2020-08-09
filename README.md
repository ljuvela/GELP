# GELP: GAN-Excited Linear Prediction

This repository contains source code and pre-trained models to reproduce the neural vocoding part in our paper "GELP: GAN-Excited Linear Prediction for Speech Synthesis from Mel-spectrogram"

The published version of the paper is freely available at https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2008.pdf

Alternatively, the paper is also available at 
https://arxiv.org/abs/1904.03976


Audio samples for copy-synthesis and Tacotron TTS are available at https://ljuvela.github.io/GELP/demopage/


### Citing

If you find the code useful, please use the following citation 
```
@inproceedings{Juvela2019,
  author={Lauri Juvela and Bajibabu Bollepalli and Junichi Yamagishi and Paavo Alku},
  title={ {GELP}: {GAN}-Excited Linear Prediction for Speech Synthesis from Mel-Spectrogram},
  year=2019,
  booktitle={Proc. Interspeech},
  pages={694--698},
  doi={10.21437/Interspeech.2019-2008},
  url={http://dx.doi.org/10.21437/Interspeech.2019-2008}
}
```

## Installing dependencies

The recommended way of handling dependencies is using Conda environments. 
Two environment versions are provided, one for CPU-only systems and another for GPUs. 

The provided model should light enough to run a copy-synthesis experiment on a laptop CPU (hence the inclusion). 

To create and activate the environment for CPU
```
conda env create -f environment-cpu.yml
conda activate gelp-cpu
``` 

Similarly, for GPU machines
```
conda env create -f environment-gpu.yml
conda activate gelp-gpu
``` 

### Note on TensorFlow versions

These scripts were originally written using TensorFlow 1.13 and from TensorFlow 1.14 and above throw deprecation warnings regarding TensorFlow 2.0. Making the scripts compatible with TF 2 is not trivial due to major changes in static vs. dynamic graphs and removal of global namespacing, etc.

If you'd like to suppress the warnings, include a `--suppress_warnings` flag in the command line arguments of the main scripts.

## Copy-synthesis with a pre-trained model

A few speech waveform files from test set are included in the repository for demonstration purposes. These are located in `demo_data/wav`.

Run the following script to extract mel-spectrogram from wav files and resynthesize speech using a pre-trained GELP model.
```
python generate_copysyn.py \
    --model_dir sessions/pretrained \
    --input_wav_dir demo_data/wav \
    --suppress_warnings
```

After running the script, you should find synthesized waveforms at `sessions/pretrained/syn` and the corresponding excitation waveforms at `sessions/pretrained/exc`.

Running synthesis again will create slightly different results, because GELP is a non-deterministic generative model (as all GANs are). 

### Mel-spectrograms and Tacotron TTS

If you would like to additionally save the mel-spectrogram, add a `--save_melspec` argument to the call and the spectrograms should appear in `sessions/pretrained/mel`. 

Although the mel-spectrogram configuration used here is close to popular Tacotron repositories (such as https://github.com/keithito/tacotron), we have experienced that subtle differences can cause serious issues with the signal processing related to mel-inversion and envelope fitting.

Partially due to these concerns, we cannot currently provide off-the-shelf usage as a generic mel-spectrum-to-waveform neural vocoder. Nevertheless, it's fairly simple to partition the `generate_copysyn.py` script into an acoustic feature extractor and a vocoder synthesis part. 

## Training a model

Training GAN models can be tedious (even when everything is stable), due to no clear stopping criterion and occasional cycles in the Generator vs. Discriminator training dynamic. Based on my experience, I recommend monitoring as many things as possible and saving snapshots of both the models and generated audio.

### Preparation

For a training toy example, let's use the wave files in `demo_data/wav`. 
First make a list of the wavefiles by calling

```bash
python make_filelist.py \
    --input_wav_dir demo_data/wav \
    --filelist_name demo_data/train_list_demo.txt
```

The default configuration file is located at 
`config/config.json`. I recommend making and editing a copy that is not under version control. 

Make sure that `data_dir` and `train_list` point to the correct locations (they should for this demo). The demo is set to train just for one iteration, feel free to change `max_iters` to a larger value for serious experiments. I've usually done 50k-200k iterations at a time and checked if the model is going anywhere.

### Tensorboard

In the default configuration, the model id is set as `1`. To start a Tensorboard for the logs run
```bash
tensorboard --logdir sessions/1/tensorboard/ 
```

Additionally, audio snapshots of generated and ground-truth speech and excitation signals are being saved in `sessions/1/audio_snapshots`.

### Adapting a pre-trained model

Starting with a pre-trained model and swapping the data is probably the best way to go when there is no need to change the configuration and the goal is to build a new voice.

This can be done by running

```bash
python train.py \
    --config config/config.json \
    --initial_model sessions/pretrained/model/model.ckpt \
    --suppress_warnings
```

### Training from scratch

Train the model from random initialization using the same script, just omit the `--initial_model` argument. If you'd like to experiment with the configuration, this is required. 

```bash
python train.py \
    --config config/config.json \
    --suppress_warnings 
```

I trained the best models by first pre-training with the losses in excitation domain 
```json
"stft_loss_domain" : "glot",
"gan_loss_domain" : "glot",
```
And after 50k iterations switching to 
```json
"stft_loss_domain" : "wave",
"gan_loss_domain" : "wave",
```

No Pain, No GAN. Good Luck! 
