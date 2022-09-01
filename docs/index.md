---
title: Demonstration on HRTF Interpolation from Spatially Sparse Measurements Using Autoencoder with Source Position Conditioning
description: Yuki Ito, Tomohiko Nakamura, Shoichi Koyama, and Hiroshi Saruwatari (The University of Tokyo)
layout: default_minimal
---

In this demo page, we show music source separation results using our proposed MRDLA [1] and conventional time-domain audio source separation methods. The mixture and ground truth signals of musical instruments (vocals, bass, drums, and other) are from the MUSDB18 dataset [2].

<!-- | Source Position | a |  Ground Truth | RLR-based method [2] | Proposed [1] |
| ---- | ---- | ---- | ---- | ---- |
|  <img src="figure/srcpos2d_000_090_top.png" width="200" height="200"> | <img src="figure/srcpos2d_000_090_side.png" width="200" height="200"> | <audio controls preload="meatadata" src="audio/sub4_azim000_zeni090_gt.wav"></audio>  | <audio controls preload="meatadata" src="audio/sub4_azim000_zeni090_rlr.wav"></audio> | <audio controls preload="meatadata" src="audio/sub4_azim000_zeni090_420428.wav"></audio> | -->

<table class="sampleTable" style="table-layout: fixed;">
  <thead>
    <tr>
      <th>Source Position</th>
      <th>a</th>
      <th>Ground Truth</th>
      <th>RLR-based method [2]</th>
      <th>Proposed [1]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="figure/srcpos2d_000_090_top.png" width="200" height="200"></td>
      <td><img src="figure/srcpos2d_000_090_side.png" width="200" height="200"></td>
      <td><audio controls="" preload="meatadata" src="audio/sub4_azim000_zeni090_gt.wav"></audio></td>
      <td><audio controls="" preload="meatadata" src="audio/sub4_azim000_zeni090_rlr.wav"></audio></td>
      <td><audio controls="" preload="meatadata" src="audio/sub4_azim000_zeni090_420428.wav"></audio></td>
    </tr>
  </tbody>
</table>




## References
[1] Yuki Ito, Tomohiko Nakamura, Shoichi Koyama, and Hiroshi Saruwatari, **"Head-Related Transfer Function Interpolation from Spatially Sparse Measurements Using Autoencoder with Source Position Conditioning,"** in *Proc. International Workshop on Acoustic Signal Enhancement (IWAENC)*, Sep., 2022. (to appear) [[PDF]](https://arxiv.org/abs/2207.10967)

[2] Ramani Duraiswami, Dmitry N. Zotkin, and Nail A. Gumerov, **“Interpolation and range extrapolation of HRTFs [head related transfer functions],”** in *Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)*, 2004, vol. 4, pp. 45–48.

[3] Fabian Brinkmann, Manoj Dinakaran, Robert Pelzer, Peter Grosche,  Daniel Voss, and Stefan Weinzierl, **"A cross-evaluated database of measured and simulated HRTFs including 3D head meshes, anthropometric features, and headphone impulse responses"**, *J. Audio Eng. Soc.*, vol. 67, no. 9, pp. 705–718, 2019.

[4] Fabian Brinkmann, Manoj Dinakaran, Robert Pelzer, Jan Joschka Wohlgemuth, Fabian Seipel, Daniel Voss, Peter Grosche, Stefan Weinzierl, **"The HUTUBS head-related transfer function (HRTF) database,"** 2019, url: http://dx.doi.org/10.14279/depositonce-8487 (accessed May 6, 2022).

[5] Zafar Rafii, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner, **"MUSDB18-HQ - an uncompressed version of MUSDB18,"** 2019, url: https://doi.org/10.5281/zenodo.3338373 (accessed Aug. 30, 2022).