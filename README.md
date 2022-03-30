# CUL

This is an offical implementation of "Contrastive Uncertainty Learning for Iris Recognition with Insufficient Labeled Samples" (Accepted by IJCB, oral).
To best of my knowledge, CUL is the **first work for unsupervised iris recognition**, and it achieves data augmentation based on the probabilistic embedding (uncertainty embedding) and applies contrastive self-supervised learning. 


## Why we developed unsupervised iris recognition?
The core of its answer is the Application scenarios of unsupervised iris recognition.

When deploying an iris recognition system in a new environment, it is easy to occur severe performance degradation. 
The performance degradation is generally caused by the gap in acquisition conditions between the deployment scene and the training scene, a.k.a, cross-database setting.
To better promote the development of iris recognition, we propose a compromise setting that closely mimics the realworld scenario, named iris recognition with insufficient labeled samples. 
In this new setting, the model can be quickly deployed in a new environment with satisfactory performance using limited labeled data and abundant unlabeled
data. In the paper, the proposed CUL utilizes partially- or un-labeled data to mitigate this performance degradation.


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

