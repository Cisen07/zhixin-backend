from flask import Flask
import cv2
from facial_emotion import FacialEmotionDetector
from audio_emotion import AudioEmotionDetector
from rPPG import HeartRate
from moviepy.editor import *

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/getinfo/<filename>')
def user(filename):
    path = 'video/' + filename
    capture = cv2.VideoCapture(path)

    # 面部情绪识别
    step_length =10  # 如果step_length为1时，是对每一帧进行操作
    facial_emotion_detector = FacialEmotionDetector()
    json0 = facial_emotion_detector.get_facial_emotion(capture, step_length)

    # 从视频中提取并保存音频信息
    audio_path = 'audio/test.wav'
    video = VideoFileClip(path)
    audio = video.audio
    audio.write_audiofile(audio_path)

    # 音频情绪识别
    audio_emotion_detector = AudioEmotionDetector()
    json1 = audio_emotion_detector.get_audio_emotion(audio_path)

    # 心率提取
    heart_rate = HeartRate()
    json2 = heart_rate.do(path)

    print(json0)
    print(json1)
    print(json2)

    return json2


if __name__ == '__main__':
    app.run()
