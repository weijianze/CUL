# Contrastive uncertainty learning (CUL)

This is an official implementation of "Contrastive Uncertainty Learning for Iris Recognition with Insufficient Labeled Samples" (Accepted by IJCB, [oral](https://ieeexplore.ieee.org/abstract/document/9484388/).
To the best of my knowledge, CUL is the **first work for unsupervised iris recognition**, and it achieves data augmentation based on probabilistic embedding (uncertainty embedding) and applies contrastive self-supervised learning. 


## Motivation: Why we develop unsupervised iris recognition?
The core of its answer is the **Application Scenarios** of unsupervised iris recognition.

When deploying an iris recognition system in a new environment, it is easy to occur severe performance degradation. 
The performance degradation is generally caused by the gap in acquisition conditions between the deployment scene and the training scene, a.k.a, cross-database setting.
To better promote the development of iris recognition, we propose a compromise setting that closely mimics the real-world scenario, named iris recognition with insufficient labeled samples (or insufficient labels). 
In this new setting, the model can be quickly deployed in a new environment with satisfactory performance using limited labeled data and abundant unlabeled
data. In the paper, the proposed CUL utilizes partially- or un-labeled data to mitigate this performance degradation.

## Method: contrastive uncertainty learning
This approach is built upon probabilistic embedding.
Thus, we introduce probabilistic embedding first.
### Probabilistic embedding
This new representation adopts a Gaussian distribution rather than the conventional deterministic point to represent an iris image.
The mean and variance of the Gaussion distribution encode the identity and uncertainty information, respectively.
Based on the probabilistic embedding, each sample point (iris features) of the iris image can be regarded as a shifted point of the class center (distribution mean/identity feature), and the shifting effect comes from the uncertainty information (distribution variance/uncertainty information).
More importantly, we can obtain more virtual points by repetitively sampling from the probabilistic embedding, which makes CUL feasible.

In addition, we want to explain why we augment data using probabilistic embedding instead of existing augmentation methods (like cutout, flip, ...).
Existing augmentation methods almost obtain diverse data in the pixel-level space.
However, for a recognition task, the augmented images of these methods maybe not meet the iris image acquisition criteria, such as ISO/IEC SC37,
ISO/IEC 19794-6 and ISO/IEC 29794-6.
It means that the augmented images would not appear in the real world.
Different from the existing pixel-level augmentation methods, the data augmentation based on probabilistic embedding is a proper augmentation in the feature-level space for iris images.
In addition, the lower computational cost of data augmentation is another advantage of our method. Since our augmentation is in the feature-level space, it avoids multiple forwards of pixel-level augmentation.


### Contrastive uncertainty learning loss
(TODO) I am being busy with graduation now, and this part will be added soon.

## Prerequisites
This implementation is based on platform of pytorch 1.7, our environment is:
- Linux
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN
- Pytorch 1.7
- Torchvision  0.8.2
- Pillow  8.1
- Numpy   1.19.5
- Scikit-learn  0.24.0
- Scipy  1.5.4
- Ipython  7.16.1
- Thop (for computational complexity)

## Recognition performance
Recognition performance on the CASIA-Thousand dataset:
| --       | FNMR@FMR | EER        | 10^{-3} | 10^{-5} |
| -------- | -------- | ---------- | ------- | ------- |
| Pretrain | 0&0      | 2.74       | 12.04   | 32.87   |
| Semi     | 1&1      | 1.44(0.36) | 4.52    | 15.94   |
|          | 1&3      | 1.42(0.39) | 4.46    | 15.11   |
|          | 1&5      | 1.23(0.25) | 3.79    | 14.19   |
|          | 1&7      | 1.17(0.19) | 3.56    | 13.36   |
|          | 1&9      | 1.10(0.04) | 3.21    | 12.04   |
| Un-      | 0&10     | 1.28       | 4.03    | 14.81   |
| Fully-    | 10&0     | 0.85       | 2.29    | 11.26   |


Recognition performance on the CASIA-Distance dataset:
| --       | FNMR@FMR | EER        | 10^{-3} | 10^{-5} |
| -------- | -------- | ---------- | ------- | ------- |
| Pretrain | 0&0      | 3.72       | 14.83   | 34.48   |
| Semi     | 1&1      | 1.72(0.10) | 5.73    | 19.64   |
|          | 1&3      | 1.70(0.08) | 5.72    | 17.77   |
|          | 1&5      | 1.68(0.08) | 5.28    | 17.25   |
|          | 1&7      | 1.65(0.14) | 5.28    | 15.34   |
|          | 1&9      | 1.57(0.12) | 5.01    | 13.78   |
| Un-      | 0&10     | 1.68       | 5.89    | 16.84   |
| Fully-    | 10&0     | 1.46       | 4.32    | 12.08   |

## Citation

```
@article{wei2022iris,
  author={Jianze Wei and 
          Huaibo Huang and
          Yunlong Wang and
          Ran He and 
          Zhenan Sun}
  title={Towards More Discriminative and Robust Iris Recognition by Learning Uncertain Factors}, 
  journal={IEEE Transactions on Information Forensics and Security}, 
  year={2022},
  volume={17},
  pages={865-879},
  publisher={IEEE}
}

```

```
@inproceedings{wei2021Contrastive,
  author    = {Jianze Wei and
               Ran He and
               Zhenan Sun},
  title     = {Contrastive Uncertainty Learning for Iris Recognition with Insufficient Labeled Samples},
  booktitle = {International {IEEE} Joint Conference on Biometrics},
  pages     = {1--8},
  publisher = {{IEEE}},
  year      = {2021},
}

