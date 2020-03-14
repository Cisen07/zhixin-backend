import tensorflow as tf
import os
import numpy as np
import keras
from keras.models import model_from_json

from Config import Config
import Opensmile_Feature as of


class AudioEmotionDetector:
    load_model_name = 'LSTM_OPENSMILE'
    model_name = 'lstm'

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # make GPU unavaliable

    def __init__(self):
        print("Audio Emotion Init...")

    def au_load_model(self, load_model_name):
        # 加载json
        print("de2")
        model_path = 'audio_emotion/Models/' + load_model_name + '.h5'
        model_json_path = 'audio_emotion/Models/' + load_model_name + '.json'

        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        print("de3")
        # 加载权重
        model.load_weights(model_path)
        print("de4")

        return model

    def au_predict(self, model, graph, file_path: str):
        file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
        # playAudio(file_path)

        of.get_data(file_path, Config.PREDICT_FEATURE_PATH_OPENSMILE, train=False)
        test_feature = of.load_feature(Config.PREDICT_FEATURE_PATH_OPENSMILE, train=False)

        test_feature = np.reshape(test_feature, (test_feature.shape[0], 1, test_feature.shape[1]))

        with graph.as_default():
            result = model.predict(test_feature)
        result = np.argmax(result)

        result_prob = model.predict_proba(test_feature)[0]

        return Config.CLASS_LABELS[int(result)], result_prob

    def get_audio_emotion(self, audio_path):
        keras.backend.clear_session()  # 在新的model加载权值前需要这个操作
        print("de1")
        model = self.au_load_model(self.load_model_name)
        graph = tf.get_default_graph()

        print("the file_path is ", audio_path)
        emo, prob = self.au_predict(model, graph, audio_path)

        # jsonR = {
        #     "audio_emotion": "%s" % emo,
        #     "probability": "%s" % max(prob)
        # }

        return emo, max(prob)
