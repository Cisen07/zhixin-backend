from threading import Thread
import sys

from heart_rate.pulse import Pulse
from heart_rate.utils import *


class ProcessMasks():
    def __init__(self, sz=270, fs=30, bs=30, size=256, heart_rate_file='heart_rate_file.csv'):  # 分别是signalsize，framesize，batchsize，imagesize
        print('init')
        self.stop = False
        self.masked_batches = []
        self.batch_mean = []
        self.signal_size = sz
        self.batch_size = bs
        self.signal = np.zeros((sz, 3))  # 默认是270*3的二维数组，3应该是红绿蓝三通道
        self.pulse = Pulse(fs, sz, bs, size)
        self.hrs = []
        self.save_results = True
        self.heart_rate_file = heart_rate_file  # 创建一个文件来存放获得的心率
        with open(self.heart_rate_file, 'w+') as hrf:  # 如果先前有这个文件，则需要新建来覆盖掉
            pass

    def __call__(self, pipe, plot_pipe, source):
        self.pipe = pipe
        self.plot_pipe = plot_pipe
        self.source = source

        compute_mean_thread = Thread(target=self.compute_mean)  # 每30帧，计算每一帧每一通道中平均bgr值，应该处在0到255之间；注意为0的那些没有人脸的部分不计数；并且要在字典中标记是否有人脸
        compute_mean_thread.start()
        
        extract_signal_thread = Thread(target=self.extract_signal)
        extract_signal_thread.start()

        self.rec_frames()  # 从获取脸部皮肤的进程处接收数据
        
        compute_mean_thread.join()
        extract_signal_thread.join()

    def rec_frames(self):
        while True and not self.stop:
            data = self.pipe.recv()

            if data is None:
                self.terminate()
                break
            batch = data[0]  # batch.shape = (480, 640, 3)；480*640是摄像头分辨率

            self.masked_batches.append(batch)

    def process_signal(self, batch_mean):
        size = self.signal.shape[0]  # signal是270*3
        b_size = batch_mean.shape[0]  # batch_mean是30*3
        # 数据存储容量是270帧，以30帧为单位存储，当存满了270帧，就把后240帧前移，覆盖最早的30帧，空出后30帧的位置，可以放入新的30帧数据
        self.signal[0:size-b_size] = self.signal[b_size:size]  # signal[0:240] = signal[30:270]
        self.signal[size-b_size:] = batch_mean  # signal[240:270] = batch_mean
        p = self.pulse.get_pulse(self.signal)  # CDF和ASF两种滤波，然后使用三通道的均值；270*3 -> 270
        p = moving_avg(p, 6)  # 移动平均滤波：去除信号中的异常点
        hr = self.pulse.get_rfft_hr(p)
        if len(self.hrs) > 300: self.hrs.pop(0)  # 大于300的心率肯定是错误的，去掉

        self.hrs.append(hr)
        # 如果不能画图，就把HeartRate的结果打印输出
        hr_fft = moving_avg(self.hrs, 3)[-1] if len(self.hrs) > 5 else self.hrs[-1]
        sys.stdout.write(f'\rHr: {round(hr_fft, 0)}')
        sys.stdout.flush()

        hr_fft = moving_avg(self.hrs, 3)[-1] if len(self.hrs) > 5 else self.hrs[-1]
        with open(self.heart_rate_file, 'a+') as hrf:  # 把心率写到csv文件中
            hrf.write(str({round(hr_fft, 0)})+'\n')

    def extract_signal(self):
        signal_extracted = 0
        
        while True and not self.stop:
            if len(self.batch_mean) == 0:  # 轮询直到batch_mean中有数据，也就是masked_batches中有30帧数据并且被compute_mean函数处理过的到平均值了
                time.sleep(0.01)
                continue
            
            mean_dict = self.batch_mean.pop(0)
            mean = mean_dict['mean']  # (30, 3)；每一帧每一通道中平均bgr值，应该处在0到255之间；注意为0的那些没有人脸的部分不计数

            if mean_dict['face_detected'] == False:  # 检测发现字典中标注这30帧无人脸
                # print("no face detected")
                if self.plot_pipe is not None:
                    self.plot_pipe.send('no face detected')
                continue
            if signal_extracted >= self.signal_size:  # 当270帧信号装载进signal了，进入process_signal函数
                self.process_signal(mean)
            else:  # eg:第一步是signal[0:30] = mean，第二步是signal[30:60] = mean……
                self.signal[signal_extracted: signal_extracted + mean.shape[0]] = mean
                # print("load signal")
            signal_extracted += mean.shape[0]

    def compute_mean(self):
        curr_batch_size = 0
        batch = None
        while True and not self.stop:
            if len(self.masked_batches) == 0:  # 以0.01秒为单位轮询检测从获取脸部皮肤识别进程处累计获得的帧数不为0
                time.sleep(0.01)
                continue
            
            mask = self.masked_batches.pop(0)     
            if batch is None:
                batch = np.zeros((self.batch_size, mask.shape[0], mask.shape[1], mask.shape[2]))  # 初始化一个形状为(30, 480, 640, 3)的tensor来放置1个batch的数据
            
            if curr_batch_size < (self.batch_size - 1):  # 累计够30帧的数据再进行下一步计算
                batch[curr_batch_size] = mask
                curr_batch_size+=1
                continue
            
            batch[curr_batch_size] = mask
            curr_batch_size = 0

            non_zero_pixels = (batch!=0).sum(axis=(1,2))  # 对batch这个张量，在480和640这两个维度进行压缩，得到一个形状为(30, 3)的张量，压缩方式是计算着480*640个元素(某一帧某一通道下)中不为0的数量
            total_pixels = batch.shape[1] * batch.shape[2]  # 480*640 = 307200
            avg_skin_pixels = non_zero_pixels.mean()  # 30*3个元素进行求平均，得到一个标量，表示平均有皮肤的像素数量
            m = {'face_detected': True, 'mean': np.zeros((self.batch_size, 3))}  # mean的形状是(30, 3)
            if (avg_skin_pixels + 1) / (total_pixels) < 0.02:  # 有皮肤的像素数量超过分辨率5%的时候视为检测到人脸；这个+1有什么意义？
                m['face_detected'] = False
            else:
                m['mean'] = np.true_divide(batch.sum(axis=(1,2)), non_zero_pixels+1e-6)  # 计算每一帧每一通道中平均bgr值，应该处在0到255之间；注意为0的那些没有人脸的部分不计数

            self.batch_mean.append(m)

    def terminate(self):
        
        # if self.plot_pipe is not None:
        #     self.plot_pipe.send(None)
        self.savePlot(self.source)
        self.saveresults()
        self.stop = True

    def saveresults(self):
        """
        saves numpy array of heart rates as hrs
        saves numpy array of power spectrum as fft_spec
        """
        np.save('hrs', np.array(self.hrs))
        np.save('fft_spec', np.array(self.pulse.fft_spec))
