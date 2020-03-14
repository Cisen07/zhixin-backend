# audio emotion recognition 参数配置


class Config:

    # 情感标签
    CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")

    # LSTM 的训练 epoch 数
    epochs = 20

    # Opensmile 标准特征集
    CONFIG = 'IS10_paraling'
    # Opensmile 安装路径
    OPENSMILE_PATH = '/media/cisen/923253EA3253D1B9/桌面快捷方式/UBUNTU/opensmile-2.3.0'
    # 每个特征集的特征数量
    FEATURE_NUM = {
        'IS09_emotion': 384,
        'IS10_paraling': 1582,
        'IS11_speaker_state': 4368,
        'IS12_speaker_trait': 6125,
        'IS13_ComParE': 6373,
        'ComParE_2016': 6373
    }

    # 特征存储路径
    FEATURE_PATH = 'audio_emotion/Features/6-category/'
    # 训练特征存储路径（Opensmile）
    TRAIN_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'train_opensmile_casia.csv'
    # 预测特征存储路径（Opensmile）
    PREDICT_FEATURE_PATH_OPENSMILE = FEATURE_PATH + 'test_opensmile_casia.csv'
    # 训练特征存储路径（librosa）
    TRAIN_FEATURE_PATH_LIBROSA = FEATURE_PATH + 'train_librosa_casia.p'
    # 预测特征存储路径（librosa）
    PREDICT_FEATURE_PATH_LIBROSA = FEATURE_PATH + 'test_librosa_casia.p'

    # 模型存储路径
    MODEL_PATH = 'audio_emotion/Models/'
