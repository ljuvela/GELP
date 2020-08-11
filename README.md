# GELP: GAN-Excited Linear Prediction

This repository contains source code and pre-trained models to reproduce the neural vocoding part in our paper "GELP: GAN-Excited Linear Prediction for Speech Synthesis from Mel-spectrogram"

The published version of the paper is freely available at https://www.isca-speech.org/archive/Interspeech_2019/abstracts/2008.html

Alternatively, the paper is also available at 
https://arxiv.org/abs/1904.03976


Audio samples for copy-synthesis and Tacotron TTS are available at https://ljuvela.github.io/GELP/demopage/


### Citing

If you find the code useful, please use the following citation 

Juvela, L., Bollepalli, B., Yamagishi, J., Alku, P. (2019) GELP: GAN-Excited Linear Prediction for Speech Synthesis from Mel-Spectrogram. Proc. Interspeech 2019, 694-698, DOI: 10.21437/Interspeech.2019-2008.

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

## Understanding the configuration

This section contains explanations of th configuration parateters contaned in `config/config.json`.

`run_id` is the base name for training a run. The id can be a number or a string and it is useful to change for each new experiment.
```json
"run_id" : 1,
```  
 
### Data configuration

Data configuration contains information about the dataset and features, including the mel-spectrogram configuration
```json
"data" : {
```   

`data_dir` points to the base data directory directory, and the path can be absolute or relative. `train_list` contains the basenames (no path or extension) of the training files, one file per line. It should be placed directly in `data_dir`. 

```json   
    "data_dir" : "demo_data",
    "train_list" : "train_list_demo.txt",
```  
`audio_dir` is a subdirectory of `data_dir` that contains the actual audio files, while `audio_ext` marks the extension.
`audio_dim` corresponds to the number of channels in audio (although only mono is supported currently). `sample_rate` should match the sample rate of the audio files. 

```json   
    "audio_dir" : "wav",
    "audio_ext": ".wav",
    "audio_dim" : 1,
    "sample_rate": 16000,
```  

`num_mels` determines the size of the mel-filterbank, while `dct_coefs` selects the number of DCT coefficients retained for MFCCs. When set to `null`, mel-spectrogram is used instead of MFCCs.  
```json
    "num_mels": 80,
    "dct_coefs": null,
``` 
`num_freq` set the number of frequency bins in STFT. For example, 1025 bins corresponds to FFT size of 2048. `frame_length_ms` and `frame_shift_ms` set the STFT frame length and hop size in milliseconds, while `preemphasis` sets the first-order differentiator coefficient used in pre-emphasis. 
```json
    "num_freq": 1025,
    "frame_length_ms": 25.0,
    "frame_shift_ms": 5.0,
    "preemphasis": 0.97,
```  
`min_level_db` and `ref_level_db` are normalization coefficients used to limit (and clip) the dynamic range of the mel-spectrograms.
```json
    "min_level_db": -100,
    "ref_level_db": 20,
``` 

`ar_filter_order` determines the order of AR model polynomial. A typical choice for text-to-speech is 30, but we used 24 in the paper for compatibility with our previous work on MFCC synthesis. 
```json 
    "ar_filter_order": 24
```  
  

### Model configuration 

The Generator `model` is the main audio-rate model in GELP. 
The width of WaveNet layers is determined by `residual_channels` and  `postnet_channels` (also known as skip channels), while `dilations` lists the dilation factor for each convolution layer and decides how many convolution layers are used. `filter_width` sets the kernel width in convolutions, and `causal` allows switching between causal and non-causal padding modes. Additionally, `add_noise_at_each_layer` introduces an independent noise source to each convolutional layer. 
```json
"model": {
    "residual_channels": 64,
    "postnet_channels": 64,
    "cond_embed_dim": 64,
    "dilations": [1, 2, 4, 8, 16, 32, 64, 128,
                  1, 2, 4, 8, 16, 32, 64, 128,
                  1, 2, 4, 8, 16, 32, 64, 128],
    "filter_width" : 5,
    "causal" : false,
    "add_noise_at_each_layer": false
},
```  

The `cond_model` model is a smaller WaveNet operating on the frame-rate acoustic features and providing conditioning for the Generator. The WaveNet configuration parameters are the same as above, with an additional parameter `n_hidden` setting the output size (injected as conditioning to the Generator).


```json
"cond_model" : {
    "residual_channels" : 64,
    "postnet_channels" : 64,
    "dilations" : [1, 2, 4, 8, 1, 2, 4, 8],
    "filter_width": 5,
    "n_hidden": [64]
},
```  
The another GAN model, `discriminator` is likewise a (non-causal) WaveNet with similar architecture parameters.
```json
"discriminator" : {
    "dilations" : [1, 2, 4, 8, 16, 32, 64,
                   1, 2, 4, 8, 16, 32, 64],
    "postnet_channels" : 64,
    "residual_channels" : 64,
    "filter_width" : 5                       
},
```  

### Training config

The training section of the config contains settings for training, losses, and logging. 
```json
"training" : {
```

`num_epochs` sets how many times the training can go through the full training set before stopping, while `max_iters` is a similar stopping term for number of iterations (i.e., minibatch parameter updates). Whichever criterion is reached first stops the training. 
```json  
    "num_epochs" : 100,
    "max_iters" : 100000,
```  

`adam_lr`, `adam_beta1` and `adam_beta2` set the learning rate and the Adam optimizer hyperparameters for all the models. Currently these are shared, but having individual settings is an easy modification.
```json
    "adam_lr" : 1e-4,
    "adam_beta1" : 0.9,
    "adam_beta2" : 0.999,
```  

`min_samples` and `max_samples` control the audio segment minimum and maximum length in samples. Specifying a range is meaningful when processing only one sample per iteration (batch-size-one)
```json
    "min_samples" : 16000,
    "max_samples" : 16000,
```

Weights for the different loss components are set in the following. `stft_loss_weight` controls the STFT magnitude loss, while `gan_loss_weight` controls the adversarial loss weight. `GP_loss_weight` is the weight of the Wasserstein GAN gradient penalty, while `R1_loss_weight` is related to R1 gradient magnitude regularization. `gan_loss_type` can be changed be changed, but all experiments were done with Wasserstein GAN (`wgan`)
```json
    "stft_loss_weight" : 10.0,
    "gan_loss_weight" : 1.0,
    "GP_loss_weight" : 10.0,
    "R1_loss_weight" : 1.0,
    "gan_loss_type" : "wgan",
```  

Losses can be evaluated in two domains: `wave` for speech directly and `glot` for the excitation signal. `stft_loss_domain` sets the domain for the STFT loss and `gan_loss_domain` for the adversarial loss.

```json
    "stft_loss_domain" : "wave",
    "gan_loss_domain" : "wave",
```  

`stft_perceptual_loss` enables a perceptual spectral masking in the STFT loss (similar to the perceptual model in CELP). `stft_loss_in_db` evaluates the STFT loss in log-magnitudes (or decibels) and `stft_loss_cancel_pre_emph` cancels pre-emphasis before loss evaluation.

```json    
    "stft_perceptual_loss" : false,
    "stft_loss_in_db" : false,
    "stft_loss_cancel_pre_emph" : false,
```  

`summary_interval` controls the interval (in iterations) of saving Tensorboard summaries, while `audio_snapshot_interval` and `model_save_interval` do the same for audio and model snapshots. 
```json
    "summary_interval" : 100,
    "audio_snapshot_interval" : 1000,
    "model_save_interval" : 10000
        
```  

