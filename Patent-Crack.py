#!/usr/bin/env python3
# coding=utf-8

"""    
    @File: Patent-Crack.py
    @Desc: 
    @Author: lv junling
    @Date Created: 2018/10/22
"""
import os
import pickle
import cv2
import numpy as np
from PIL import Image


class PatentCrack(object):
    def __init__(self, pkl_fn=None):
        if pkl_fn is None:
            print('[error]Must specify the pickle filename.')
            return
        self.pkl_fn = pkl_fn
        if os.path.exists(pkl_fn):
            self._load_pkl()
        else:
            self.gen_pkl_fn()

    def gen_pkl_fn(self):
        imgs_path = u'./data'
        chi_1_imgs = ['1.jpeg', '2.jpeg', '3.jpeg', '4.jpeg', '5.jpeg',
                      '6.jpeg', '7.jpeg', '8.jpeg', '9.jpeg']

        chi_2_imgs = ['10.jpeg', '11.jpeg', '12.jpeg', '13.jpeg', '14.jpeg', '15.jpeg',
                      '16.jpeg', '17.jpeg', '18.jpeg', '19.jpeg']

        op_imgs    = ['1.jpeg', '7.jpeg']

        chi_3_imgs = ['100.jpeg', '101.jpeg', '102.jpeg', '103.jpeg', '104.jpeg', '105.jpeg',
                      '106.jpeg', '107.jpeg', '108.jpeg', '109.jpeg']

        chi_1_arr = np.zeros([10, 20, 11], dtype=np.bool)
        for idx, img_fn in enumerate(chi_1_imgs):
            c1, _, _, _ = self._get_split_img(os.path.join(imgs_path, img_fn))
            chi_1_arr[idx+1] = c1

        chi_2_arr = np.zeros([10, 20, 9], dtype=np.bool)
        for idx, img_fn in enumerate(chi_2_imgs):
            _, c2, _, _ = self._get_split_img(os.path.join(imgs_path, img_fn))
            chi_2_arr[idx] = c2

        op_arr = np.zeros([3, 20, 12], dtype=np.bool)
        for idx, img_fn in enumerate(op_imgs):
            _, _, op, _ = self._get_split_img(os.path.join(imgs_path, img_fn))
            op_arr[idx] = op

        chi_3_arr = np.zeros([10, 20, 10], dtype=np.bool)
        for idx, img_fn in enumerate(chi_3_imgs):
            _, _, _, c3 = self._get_split_img(os.path.join(imgs_path, img_fn))
            chi_3_arr[idx] = c3

        fout = open(self.pkl_fn, 'wb')
        data = {'chi_1': chi_1_arr, 'chi_2': chi_2_arr, 'op': op_arr, 'chi_3': chi_3_arr}
        pickle.dump(data, fout)
        fout.close()

        self._load_pkl()

    def _load_pkl(self):
        data = pickle.load(open(self.pkl_fn, 'rb'))
        self.chi_1_arr = data['chi_1']
        self.op_arr = data['op']
        self.chi_2_arr = data['chi_2']
        self.chi_3_arr = data['chi_3']

    @staticmethod
    def _get_split_img(img_fn):
        img_arr = np.array(Image.open(img_fn).convert('L'))
        img_arr[img_arr < 156] = 1
        img_arr[img_arr >= 156] = 0
        img_arr = img_arr.astype(np.bool)
        chi_1_arr = img_arr[:,  6:17]
        chi_2_arr = img_arr[:, 19:28]
        op_arr    = img_arr[:, 32:44]
        chi_3_arr = img_arr[:, 45:55]
        return chi_1_arr, chi_2_arr, op_arr, chi_3_arr

    @staticmethod
    def _cal_result(num1, num2, num3,op):
        if op == 0:
            return num1*10 + num2 + num3
        elif op == 1:
            return num1*10 + num2 - num3
        elif op == 2:
            return num1 * num2
        else:
            return int(num1 / num2)

    def feed(self, img_fn):
        chi_1_arr, chi_2_arr, op_arr, chi_3_arr = self._get_split_img(img_fn)
        chi_1_arr = np.tile(chi_1_arr[np.newaxis, :], [10, 1, 1])
        op_arr = np.tile(op_arr[np.newaxis, :], [3, 1, 1])
        chi_2_arr = np.tile(chi_2_arr[np.newaxis, :], [10, 1, 1])
        chi_1_sum = np.sum(
            np.sum(np.bitwise_and(chi_1_arr, self.chi_1_arr), axis=2), axis=1)
        chi_2_sum = np.sum(
            np.sum(np.bitwise_and(chi_2_arr, self.chi_2_arr), axis=2), axis=1)
        op_sum = np.sum(
            np.sum(np.bitwise_and(op_arr, self.op_arr), axis=2), axis=1)
        op_sum[1] += 1   # 区分减号和加号
        chi_3_sum = np.sum(
            np.sum(np.bitwise_and(chi_3_arr, self.chi_3_arr), axis=2), axis=1)
        num1 = chi_1_sum.argmax()
        num2 = chi_2_sum.argmax()
        op = op_sum.argmax()
        num3 = chi_3_sum.argmax()
        result = self._cal_result(num1, num2,num3, op)
        print (result)


def test():
    crack = PatentCrack('Patent.pkl')

    crack.feed(os.path.join('86-0.jpeg'))
    # fn_list = [fn for fn in os.listdir(u'../04_data/企业证书/cnca')]
    # fn_list.sort()
    # for fn in fn_list[:50]:
    #     crack.feed(os.path.join(u'../04_data/企业证书/cnca', fn))


if __name__=='__main__':
    test()