import multiprocessing as mp
import numpy as np
import time
import re

from heart_rate.capture_frames import CaptureFrames
from heart_rate.process_mask import ProcessMasks


class HeartRate:

    batch_size = 30
    frame_rate = 25
    signal_size = 270
    plot_pipe = False

    heart_rate_file = "heart_rate_file.csv"  # 创建一个文件来存放获得的心率

    def __init__(self):
        print("rPPG Init...")

    def get_heart_rate(self, file_path):
        time1 = time.time()

        mask_process_pipe, chil_process_pipe = mp.Pipe()  # 心率提取进程和获取脸部皮肤的进程之间可以双向通信

        process_mask = ProcessMasks(self.signal_size, self.frame_rate, self.batch_size, self.heart_rate_file)
        mask_processer = mp.Process(target=process_mask, args=(chil_process_pipe, self.plot_pipe, file_path,), daemon=True)
        mask_processer.start()  # 心率提取进程

        capture = CaptureFrames(self.batch_size, file_path, show_mask=True)
        capture(mask_process_pipe, file_path)  # 获取脸部皮肤的进程

        mask_processer.join()

        time2 = time.time()  # 为什么要连续time两次
        print(f'time {time2 - time1}')
        return 1

    def cal_hr(self, file_path):
        hrs = []
        with open(file_path, 'r') as file:
            for line in file:
                mid = re.sub(u'{', '', line)
                hr = re.sub(u'}', '', mid)
                hrs.append(float(hr))
        return np.median(hrs)

    def do(self, file_path):
        hr_flag = self.get_heart_rate(file_path)
        if hr_flag:
            hr = self.cal_hr(self.heart_rate_file)
            # jsonR = {
            #     "heart_rate": hr
            # }
            return hr
        else:
            return 0
