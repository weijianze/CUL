# -*- coding: utf-8 -*-
"""
configuration of iris test
Bigliu20210305
"""

class Config(object):
    def __init__(self):
        self._root_path             = ['./']
        self._train_list_labeled    = ['./semi_protocol/Session_1/Whole_1_1_labeled.txt']
        self._train_list_unlabeled  = ['./semi_protocol/Session_1/Whole_1_1_unlabeled.txt']
        self._num_class             = 902 # #classes for labeled training data

        self._root_path_test        = ['../../08CASIA-Iris-Thousand/']
        self._test_list             = ['../../08CASIA-Iris-Thousand/test.txt']

        self.data_name              = 'CASIA IrisV4-thousand'
        self.test_type              = 'Within'
        self.semi_type              = 'Whole'

    def num_classGet(self):
        return self._num_class

    def load_detailGet_labeled(self):
        return self._root_path, self._train_list_labeled

    def load_detailGet_unlabeled(self):
        return self._root_path, self._train_list_unlabeled
    
    def test_loaderGet(self):
        return self._root_path_test, self._test_list