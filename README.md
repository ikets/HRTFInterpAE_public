# HRTFInterpAE
**[Paper](https://arxiv.org/abs/2207.10967) | [Slides](https://github.com/ikets/HRTFInterpAE_public/blob/main/docs/Ito_IWAENC2022_public.pdf) | [Demo](https://ikets.github.io/HRTFInterpAE_public/)**

<img src="https://github.com/ikets/HRTFInterpAE_public/blob/main/docs/figure/spatial_upsampling.png">

This repository contains the implementation of **"Head-Related Transfer Function Interpolation from Spatially Sparse Measurements Using Autoencoder with Source Position Conditioning"** presented in IWAENC2022.
Please cite [1] in your work when using this code in your experiments.

## Requirements
We checked the code with the following computational environment.
- Ubuntu 20.04.2 LTS
- GeForce RTX 3090 (24GB VRAM)
- Python 3.8.10
  ```
  matplotlib==3.4.2
  numpy==1.21.0
  python-sofa==0.2.0
  scipy==1.7.0
  tensorboard==2.6.0
  torch==1.9.0+cu111
  torchaudio==0.9.0
  ```
## Tutorial on Colab
You can test our pretrained model with [a short tutorial notebook](https://github.com/ikets/HRTFInterpAE_public/blob/main/tutorial.ipynb) we have prepared.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ikets/HRTFInterpAE_public/blob/main/tutorial.ipynb)
## Installation
1. Clone this repository.
2. Ensure that your computational environment meets the above or similar requirements.
  You can use `conda` and [`hrtfinterpae.yml`](https://github.com/ikets/HRTFInterpAE_public/blob/main/hrtfinterpae.yml) to quickly build an environment.
    ```
    conda env create -n hrtfinterpae -f hrtfinterpae.yml
    conda activate hrtfinterpae
    ```
3. Download [the HUTUBS HRTF database](http://dx.doi.org/10.14279/depositonce-8487) [2,3] and extract `HRIRs.zip`.
4. Locate the directories as shown below.
    ```
    ***/
      ├ HRTFInterpAE_public/
      └ HUTUBS/
        └HRIRs/
    ```
## Test
To test a pretrainded model,
  1. Move to `HRTFInterpAE_public`.<br>
      ```
      cd ***/***/HRTFInterpAE_public
      ```
  2. Make a directory which contains output files. <br>
      ```
      mkdir -p outputs/model_0_test
      ```
  3. Test the model. <br>
      ```
      nohup python3 -u main.py -c 0 -a outputs/model_0_test -test -lf 'outputs/model_0/hrtf_interp_ae.best.net' > 'outputs/model_0_test/log_test.txt' &
      ```
## Train
To train our model with the HUTUBS database, 
  1. Move to `HRTFInterpAE_public`.<br>
      ```
      cd ***/***/HRTFInterpAE_public
      ```
  2. Make a directory which contains output files. <br>
      ```
      mkdir -p outputs/model_0_train
      ```
  3. Train a model. <br>
      ```
      nohup python3 -u main.py -c 0 -a outputs/model_0_train > 'outputs/model_0_train/log.txt' &
      ```
  - The log is written in `outputs/model_0_train/log.txt`.
  - You can see loss graphs in tensorboard by 
      ```
      tensorboard --logdir outputs/model_0_train/logs/***
      ```
  - The test will start automatically after the training.

To train a model with a **new configuration**,
  1. Create a new dict `config_*` in `configs.py`. <br>
  `*` must be an integer.
  2. With the option `-c *`, run `main.py` in the same way as above.
- For other command-line arguments, see `arg_parse()` in `main.py`.

## Note
Files in `t_des` are 3D Cartesian coordinates of the points contained in spherical t-design [4] and are obtained from [5].

## Cite
```
@inproceedings{Ito:IWAENC2022,
  author    = {Yuki Ito and Tomohiko Nakamura and Shoichi Koyama and Hiroshi Saruwatari},
  title     = {Head-Related Transfer Function Interpolation from Spatially Sparse Measurements Using Autoencoder with Source Position Conditioning},
  booktitle = {in Proceedings of International Workshop on Acoustic Signal Enhancement},
  month     = {Sep.},
  year      = {2022},
  note      = {(accepted)}
}
```

## License
[CC-BY-4.0](https://github.com/ikets/HRTFInterpAE_public/blob/main/LICENSE)

## References
[1] Yuki Ito, Tomohiko Nakamura, Shoichi Koyama, and Hiroshi Saruwatari, <strong>“Head-Related Transfer Function Interpolation from Spatially Sparse Measurements Using Autoencoder with Source Position Conditioning,”</strong> in <em>Proc. International Workshop on Acoustic Signal Enhancement (IWAENC)</em>, Sep., 2022. (to appear) [[PDF]](https://arxiv.org/abs/2207.10967) [[Slides]](https://github.com/ikets/HRTFInterpAE_public/blob/main/docs/Ito_IWAENC2022_public.pdf) <br>

[2] Fabian Brinkmann, Manoj Dinakaran, Robert Pelzer, Peter Grosche,  Daniel Voss, and Stefan Weinzierl, “A cross-evaluated database of measured and simulated HRTFs including 3D head meshes, anthropometric features, and headphone impulse responses”, <em>J. Audio Eng. Soc.</em>, vol. 67, no. 9, pp. 705–718, 2019.<br>

[3] Fabian Brinkmann, Manoj Dinakaran, Robert Pelzer, Jan Joschka Wohlgemuth, Fabian Seipel, Daniel Voss, Peter Grosche, Stefan Weinzierl, “The HUTUBS head-related transfer function (HRTF) database,” 2019, url: http://dx.doi.org/10.14279/depositonce-8487 (accessed May 6, 2022).<br>

[4] X. Chen and R. S. Womersley, “Existence of solutions to
systems of underdetermined equations and spherical designs,”
<em>SIAM J. Numer. Anal.,</em> vol. 44, no. 6, pp. 2326–2341, 2006.

[5] https://www.polyu.edu.hk/ama/staff/xjchen/sphdesigns.html (accessed Sep. 18, 2022)

