import numpy as np
from heart_rate.cdf import CDF
from heart_rate.asf import ASF
from numpy.linalg import inv

from scipy.fftpack import rfftfreq, rfft
import cv2
from torchvision import transforms
import pdb
from PIL import Image

PRE_STEP_ASF = False  
PRE_STEP_CDF = False


class Pulse():
    def __init__(self, framerate, signal_size, batch_size, image_size=256):  # 30,270,30,256
        self.framerate = framerate  # 25
        self.signal_size = signal_size  # 270
        self.batch_size = batch_size  # 30
        self.minFreq = 0.9  # 最小频率
        self.maxFreq = 3  # 最大频率
        self.fft_spec = []  # ？转换成点值表示法后的不同频率的正余弦函数
        
    def get_pulse(self, mean_rgb):  # mean_rgb:(270, 3)
        seg_t = 3.2  # ？
        l = int(self.framerate * seg_t)  # 25*3.2=80 这是滑框的长度
        H = np.zeros(self.signal_size)  # 270
        B = [int(0.8 // (self.framerate / l)), int(4 // (self.framerate / l))]  # [2, 12]
        for t in range(0, (self.signal_size - l + 1)):  # 执行190次
            # pre processing steps
            C = mean_rgb[t:t+l,:].T  # C.shape=(3, 80)

            if PRE_STEP_CDF:
                C = CDF(C, B)
           
            if PRE_STEP_ASF:
                C = ASF(C)
           
            # POS Plane-Orthogonal-to-Skin
            mean_color = np.mean(C, axis=1)  # 3通道的面部rgb值求平均，得到一个80维向量（没有任何一个元素为0）
            diag_mean_color = np.diag(mean_color)  # 以这80个值作为对角线上的值，生成一个对角矩阵
            diag_mean_color_inv = np.linalg.inv(diag_mean_color)  # 矩阵求逆（对角矩阵的逆矩阵就是每个元素都变成原先的倒数）
            Cn = np.matmul(diag_mean_color_inv,C)  # Cn.shape: (3, 80)；从前几行的步骤可见，得到的值应该是分布在1左右
            projection_matrix = np.array([[0,1,-1],[-2,1,1]])  # 接下来几步不太明白了？
            S = np.matmul(projection_matrix,Cn)
            std = np.array([1, np.std(S[0,:])/np.std(S[1,:])])
            P = np.matmul(std,S)
            H[t:t+l] = H[t:t+l] + (P-np.mean(P))
        return H

    def get_rfft_hr(self, signal):
        signal_size = len(signal)
        signal = signal.flatten()  # flatten前后的shape都是256
        fft_data = np.fft.rfft(signal)  # 计算一维离散傅里叶变换以获得实际输入；从时域到频域；256个实数转换为133个复数？
        fft_data = np.abs(fft_data)

        freq = np.fft.rfftfreq(signal_size, 1./self.framerate)  # Frequency 得到频谱图的横坐标的数字频率；133个实数

        inds = np.where((freq < self.minFreq) | (freq > self.maxFreq) )[0]
        fft_data[inds] = 0  # 低于最低频率和高于最高频率的部分，振幅都设为0
        bps_freq=60.0*freq
        max_index = np.argmax(fft_data)  # 得到最大值对应索引
        fft_data[max_index] = fft_data[max_index]**2  # 最大振幅的振幅平个方？为什么要这么做呢
        self.fft_spec.append(fft_data)
        HR = bps_freq[max_index]
        return HR
