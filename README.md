# CUL

This is an offical implementation of "Contrastive Uncertainty Learning for Iris Recognition with Insufficient Labeled Samples" (Accepted by IJCB, oral).
To best of my knowledge, CUL is the **first work for unsupervised iris recognition**, and it achieves data augmentation based on the probabilistic embedding (uncertainty embedding) and applies contrastive self-supervised learning. 


## Motivation: Why we developed unsupervised iris recognition?
The core of its answer is the **Application Scenarios** of unsupervised iris recognition.

When deploying an iris recognition system in a new environment, it is easy to occur severe performance degradation. 
The performance degradation is generally caused by the gap in acquisition conditions between the deployment scene and the training scene, a.k.a, cross-database setting.
To better promote the development of iris recognition, we propose a compromise setting that closely mimics the realworld scenario, named iris recognition with insufficient labeled samples. 
In this new setting, the model can be quickly deployed in a new environment with satisfactory performance using limited labeled data and abundant unlabeled
data. In the paper, the proposed CUL utilizes partially- or un-labeled data to mitigate this performance degradation.

## Method
This approach is built upon the probabilistic embedding.
Thus, we introduce probabilistic embedding first.
### Probabilistic embedding
This new representation adopts a Gaussian distribution rather than the conventional deterministic point to represent an iris image.
The mean and variance of the Gaussion distribution present the identity and uncertainty information, respectively.
Based on the probabilistic embedding, each sample point (iris features) of the iris image can be regarded as a shifted point of the class center (distribution mean/identity feature), and the comes from the uncertainty information (distribution variance/uncertainty information).
###


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

