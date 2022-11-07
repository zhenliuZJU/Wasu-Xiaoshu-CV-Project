from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
import numpy as np
import cv2 as cv
import face_net.HseNet.facial_emotions as Hse


class face_processing():
    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=1)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mtcnn = MTCNN(
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )
        # TinyAge=torch.load('TinyAge.pt', map_location=torch.device('cpu')).to(device)
        self.resnet = torch.load('./face_net/result_epochs_8.pt', map_location=torch.device('cpu'))['model'].to(device)
        self.HSENet = Hse.HSEmotionRecognizer(device=device)
        self.face_list = torch.load('./face_net/result_epochs_8.pt', map_location=torch.device('cpu'))['classes']
        # face_list = ["Ding JiZheng", "He LongJie", "Hu Yingdong", "Li Zhao"]
        self.rank = torch.Tensor([i for i in range(101)]).to(device)

    def face_process(self, img_array):
        boxes, probs, points = self.mtcnn.detect(img_array, landmarks=True)
        face, _ = self.mtcnn(img_array, save_path='./face_net/face.png', return_prob=True)

        emotion = self.emotion_extract(face)
        identity = self.identity_classification(face)
        age = self.age_estimation(face)

        return identity, emotion, age

    def identity_classification(self, face):
        return None

    def emotion_extract(self, face):
        emotion, _ = self.HSENet.predict_emotions(np.uint8(255 * face.permute(1, 2, 0).numpy()))
        return emotion

    def age_estimation(self, face):
        return None