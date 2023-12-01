# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:19:43 2023

@author: ASH
"""
import cv2
import numpy as np
import time

epsilon = 1
win_size = 3

def k(i: int, r: int, c: int, w: int, win_size: int):
    """
        Lk 到 L 的位置映射关系
        Args:
            i: 窗口的第i个元素索引
            r: 窗口的行索引
            c: 窗口的列索引
            w: 图片宽度
    """
    r_win = i // win_size
    c_win = i - r_win * win_size
    ki = (r + r_win) * w + c + c_win
    return ki

def Gk_func(row_idx: int, col_idx: int, img: np.ndarray, win_size: int):
    '''
        计算每个窗口的Gk
        Args:
            row_idx: 窗口左上角的行索引
            col_idx: 窗口左上角的列索引
            img:     图片对象
            win_size: 窗口尺寸
    '''
    t, b, l, r = row_idx, row_idx + win_size, col_idx, col_idx + win_size
    patch = img[t:b, l:r]
    patch = patch.reshape(-1, 3)
    Gk = np.zeros((win_size * (win_size+1), 4))
    Gk[0:win_size * win_size, 0:3] = patch
    Gk[0:win_size * win_size, 3] = 1
    tail=np.diag(np.full(win_size, np.sqrt(epsilon)))
    Gk[win_size * win_size: win_size * (win_size + 1), 0:3] = tail
    return Gk

def Lk_func(gk: np.ndarray, win_size: int):
    temp=np.matmul(gk.transpose(), gk)
    temp=np.linalg.inv(temp)
    temp = np.matmul(gk, temp)
    temp = np.matmul(temp, gk.transpose())
    I = np.diag(np.full(len(gk),1))
    gk_bar = temp - I
    lk = np.matmul(gk_bar.transpose(), gk_bar)
    lk = lk[0: win_size ** 2, 0: win_size ** 2]
    return lk

def L_func(img: np.ndarray, win_size: int):
    h, w = img.shape[:2]
    L = np.zeros((h * w, h * w))
    t = time.time()
    for row in range(h - win_size + 1):
        for col in range(w - win_size + 1):
            gk = Gk_func(row, col, img, win_size)
            lk = Lk_func(gk, win_size) 

            for i in range(win_size ** 2):
                ki = k(i, row, col, w, win_size)
                for j in range(win_size ** 2):
                    kj = k(j, row, col, w, win_size)
                    L[ki, kj] += lk[i, j]
    print(f'coast:{time.time() - t:.8f}s')           
    return L

if __name__ == '__main__':
    img = cv2.imread('./Fig/2.png')
    L_func(img,3)