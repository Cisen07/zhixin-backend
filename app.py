from flask import Flask
from flask import request, jsonify
from moviepy.editor import *
import cv2

from upload_file import upload_file
from facial_emotion import FacialEmotionDetector
from audio_emotion import AudioEmotionDetector
from rPPG import HeartRate

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    size =  request.content_length
    print(file)
    if not file:
        print("No File to Upload.")
        return jsonify(
            status="No File to Upload."
        )
    elif file:
        upload_file(file)
        '''
        try:
            upload_file(file)
        except:
            print("Failed to upload. Some error occured.")
            return jsonify(
                status="Failed to upload. Some error occured."
            )
        '''
    return jsonify(
        filename=file.filename,
        status="Uploaded"
    )


@app.route('/getinfo/<filename>')
def user(filename):
    path = 'video/' + filename
    capture = cv2.VideoCapture(path)

    # 面部情绪识别
    step_length =10  # 如果step_length为1时，是对每一帧进行操作
    facial_emotion_detector = FacialEmotionDetector()
    facial_emotion, facial_probability = facial_emotion_detector.get_facial_emotion(capture, step_length)

    # 从视频中提取并保存音频信息
    audio_path = 'audio/test.wav'
    video = VideoFileClip(path)
    audio = video.audio
    audio.write_audiofile(audio_path)

    # 音频情绪识别
    audio_emotion_detector = AudioEmotionDetector()
    audio_emotion, audio_probability = audio_emotion_detector.get_audio_emotion(audio_path)

    # 心率提取
    heart_rate = HeartRate()
    heart_rate = heart_rate.do(path)

    print(facial_emotion, facial_probability, audio_emotion, audio_probability, heart_rate)

    jsonR = {
        'facial_emotion': facial_emotion,
        'facial_probability': str(facial_probability),
        'audio_emotion': audio_emotion,
        'audio_probability': str(audio_probability),
        'heart_rate': str(heart_rate)
    }
    return jsonR


if __name__ == '__main__':
    app.run()
