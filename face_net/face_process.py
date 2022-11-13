from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
import numpy as np
import cv2 as cv
from PIL import Image
from torchvision import datasets, transforms
import face_net.HseNet.facial_emotions as Hse


class face_processing():
    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=1)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.mtcnn = MTCNN(
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        # TinyAge=torch.load('TinyAge.pt', map_location=torch.device('cpu')).to(device)
        self.resnet = torch.load('./face_net/result_epochs_8.pt', map_location=torch.device('cpu'))['model'].to(device)
        self.HSENet = Hse.HSEmotionRecognizer(device=device)
        self.TinyAge = torch.load('./face_net/TinyAge.pt', map_location=torch.device('cpu')).to(device)
        self.face_list = torch.load('./face_net/result_epochs_8.pt', map_location=torch.device('cpu'))['classes']
        # face_list = ["Ding JiZheng", "He LongJie", "Hu Yingdong", "Li Zhao"]
        self.rank = torch.Tensor([i for i in range(101)]).to(device)

    def face_process(self, img_array):
        # boxes, probs, points = self.mtcnn.detect(img_array, landmarks=True)
        face, _ = self.mtcnn(img_array, save_path='./face_net/face.png', return_prob=True)
        face = np.uint8(cv.normalize(face.permute(1, 2, 0).numpy(), None, 0, 255, cv.NORM_MINMAX))

        emotion = self.emotion_extract(face)
        identity = self.identity_classification(face)
        age = self.age_estimation(face)

        return identity, emotion, age

    def identity_classification(self, face):
        return None

    def emotion_extract(self, face):
        emotion, _ = self.HSENet.predict_emotions(face)
        return emotion

    def age_estimation(self, face):
        face_age = self.transform(Image.fromarray(face))
        out_age = self.TinyAge(face_age.unsqueeze_(0).to(self.device))
        age = torch.sum(out_age * self.rank, dim=1).item()
        return age