import cv2 as cv
from keras.models import load_model
import numpy as np
import tensorflow as tf
import os


class FacialEmotionDetector:
    rect_color = (0, 255, 0)
    Labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    dataLength = 7

    # 地址设置
    haar_path = 'facial_emotion/models/haarcascade_frontalface_alt2.xml'
    path_emotion_test = 'facial_emotion/models/test.h5'
    path_emotion_final = 'facial_emotion/models/emotion-final.h5'

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # make GPU unavaliable

    # 加载头像检测分类器xml
    faceCascade = cv.CascadeClassifier(haar_path)

    # 加载表情分类器
    model = load_model(path_emotion_final)

    graph = tf.get_default_graph()

    def __init__(self):
        print("Face Emotion Init...")

    # 表情检测函数
    def predict_emotion(self, face_image_gray):
        resized_img = cv.resize(face_image_gray, (48, 48), interpolation=cv.INTER_AREA)
        image = resized_img.reshape(1, 48, 48, 1)
        with self.graph.as_default():
            ret = self.model.predict(image)
        return ret

    # 定义检测人脸的函数
    def get_face(self, pic):
        ans = self.faceCascade.detectMultiScale(
            pic,
            scaleFactor=1.1,  # 缩小比例
            minNeighbors=5,  # 指定每个候选矩阵至少包含的邻近元素个数，也就是要被检测到多少次才能算
            minSize=(30, 30),  # 最小可能的对象的大小
        )
        return ans

    # 把人脸剪切下来
    def cut_face(self, now_pic_ans, pic_origin, type):
        if len(now_pic_ans) > 0:
            for detect_Face in now_pic_ans:
                x, y, w, h = detect_Face
                Face = pic_origin[y:y + h, x:x + w]
                Face = cv.cvtColor(Face, cv.COLOR_BGR2GRAY)
                Emotion_Data = self.predict_emotion(Face)
                # print("CL.........................")
                return self.Labels[np.argmax(Emotion_Data)], np.max(Emotion_Data)

    def do(self, pic, type):
        gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)  # 图像灰化

        get = self.get_face(gray)

        emotion, probability = self.cut_face(get, pic, type)
        print("emotion: ", emotion)
        print("probability: ", probability)
        return emotion, probability

    def get_facial_emotion(self, capture, step_length=3):
        # fps = capture.get(cv.CAP_PROP_FPS)  # 帧率，意思是每一秒刷新图片的数量
        frames = capture.get(cv.CAP_PROP_FRAME_COUNT)  # 一整段视频中总的图片数量

        dict = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
        count = 0  # 计数总共预测了几帧

        for i in range(0, int(frames), step_length):
            ret, frame = capture.read()
            # cv2.imwrite("pic/%s-%d.jpg" % (filename, i), frame)
            cv.imwrite("pic/test.jpg", frame)
            img = cv.imread("pic/test.jpg")
            emotion, probability = self.do(img, 0)
            dict[emotion] += probability
            count += 1

        if count:
            facial_emotion = max(dict, key=dict.get)  # 获得字典中value最大的key

            jsonR = {
                "facial_emotion": "%s" % facial_emotion,
                "probability": "%s" % str(float(dict[facial_emotion]) / count)
            }

            return jsonR
        else:
            return 0
