import cv2
import numpy as np
import torch
from torch import nn
from heart_rate.models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import time
import sys


class CaptureFrames():

    def __init__(self, bs, source, show_mask=False):  # batch size 30
        self.frame_counter = 0
        self.batch_size = bs  # 这个batch size并没有在这个类中被使用到
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        # self.model.load_state_dict(torch.load('linknet.pth', map_location='cuda:0'))
        self.model.load_state_dict(torch.load('heart_rate/linknet.pth', map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to(self.device)
        self.show_mask = show_mask
        
        
    def __call__(self, pipe, source):
        self.pipe = pipe
        self.capture_frames(source)
  

    def capture_frames(self, source):
        
        img_transform = transforms.Compose([
            transforms.Resize((256,256)),  # 图片的大小一律调成256*256
            transforms.ToTensor(),  # 把照片转换成torch.FloatTensor的格式，0-255的值会映射到0-1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 为什么要标准化到这个值？
        ])
    
        camera = cv2.VideoCapture(source)
        time.sleep(1)
        self.model.eval()  # 主要是固定住BN层的值
        (grabbed, frame) = camera.read()  # 第一个参数True表示有获取到图片，第二个参数就是一帧图像

        time_1 = time.time()
        self.frames_count = 0
        while grabbed:
            (grabbed, orig) = camera.read()
            if not grabbed:
                continue
            
            shape = orig.shape[0:2]  # orig.shape = (480, 640, 3)；这个480*640应该是我摄像头的分辨率
            frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)  # 因为cv2默认的是bgr顺序，这里调换成rgb
            frame = cv2.resize(frame,(256,256), cv2.INTER_LINEAR )
            
            k = cv2.waitKey(1)  # 等待1秒，返回这一秒钟键盘键入对应的ASCII值，如果没有键入，默认-1
            if k != -1:
                self.terminate(camera)
                break

            a = img_transform(Image.fromarray(frame))  # 实现array到image的转换，然后使用本函数开头定义的转换方式进行处理
            a = a.unsqueeze(0)  # 在首个维度前增加一个维度，从[3, 256, 256]变成[1, 3, 256, 256]
            imgs = Variable(a.to(dtype=torch.float, device=self.device))
            pred = self.model(imgs)
            
            pred= torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])  # 插值上采样
            mask = pred.data.cpu().numpy()  # 把Variable中的tensor取出来，放在CPU中，转换成numpy格式
            mask = mask.squeeze()  # 移除数组中维度为1的维度，从(1, 1, 480, 640)到(480, 640)
            
            # im = Image.fromarray(mask)
            # im2 = im.filter(ImageFilter.MinFilter(3))
            # im3 = im2.filter(ImageFilter.MaxFilter(5))
            # mask = np.array(im3)
            
            mask = mask > 0.8
            orig[mask==0]=0  # 回显摄像头结果给心率提取进程前，把检测不为皮肤的部分都设为0，即黑色
            self.pipe.send([orig])  # 去除非皮肤部分的信息后，传给心率提取进程

            if self.show_mask:
                pass
                # cv2.imshow('mask', orig)  # 在窗口中显示出去掉非皮肤后的结果
            
            if self.frames_count % 30 == 29:
                time_2 = time.time()
                sys.stdout.write(f'\rFPS: {30/(time_2-time_1)}')  # FPS是画面每秒传输帧数
                sys.stdout.flush()
                time_1 = time.time()


            self.frames_count+=1

        self.terminate(camera)

    
    def terminate(self, camera):
        self.pipe.send(None)
        cv2.destroyAllWindows()
        camera.release()
        