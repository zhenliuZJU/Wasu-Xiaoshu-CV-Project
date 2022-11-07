from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, trainingimport torchfrom torch.utils.data import DataLoader, SubsetRandomSamplerfrom torch import optimfrom torch.optim.lr_scheduler import MultiStepLRfrom torch.utils.tensorboard import SummaryWriterfrom torchvision import datasets, transformsimport numpy as npimport osimport cv2 as cvimport timeimport HseNet.facial_emotions as Hsesoftmax = torch.nn.Softmax(dim=1)device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')mtcnn = MTCNN(    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,    device=device)TinyAge=torch.load('TinyAge.pt', map_location=torch.device('cpu')).to(device)resnet = torch.load('result_epochs_8.pt', map_location=torch.device('cpu'))['model'].to(device)HSENet = Hse.HSEmotionRecognizer(device=device)# face_list = torch.load('result_epochs_8.pt', map_location=torch.device('cpu'))['classes']face_list = ["Ding JiZheng", "He LongJie", "Hu Yingdong", "Li Zhao"]rank = torch.Tensor([i for i in range(101)]).to(device)font = cv.FONT_HERSHEY_SIMPLEXdef video_demo():    t = time.time()    # 0是代表摄像头编号，只有一个的话默认为0    cap = cv.VideoCapture(0)    if not cap.isOpened():        print("Cannot open camera")        exit()    while (True):        t_old = t        t = time.time()        fps = 1/(t-t_old)        ref, frame = cap.read()        boxes, probs, points = mtcnn.detect(frame, landmarks=True)        face, _ = mtcnn(frame, save_path='face.png', return_prob=True)        if face is not None:            emotion, _ = HSENet.predict_emotions(np.uint8(255*face.permute(1,2,0).numpy()))            out_age = TinyAge(np.uint8(255*face.permute(1,2,0).numpy()))            age = torch.sum(out_age * rank, dim=1).item()            output = resnet(torch.tensor([face.tolist()]).to(device))            identity = torch.argmax(output.data, dim=1)[0].item()            prob = softmax(output)[0][identity]            identity = face_list[identity]            # prob=0.5        if boxes is not None:            for i, box in enumerate(boxes):                frame = cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)                if prob>0.05:                    frame = cv.putText(frame, '{} {:.3f} || emotion: {} || age: {}'.format(identity, prob.item(), emotion, age), (0, 30), font, 0.5, (0, 255, 0), 2)                    frame = cv.putText(frame, '{:.3f}'.format(fps),                                       (0, 50), font, 0.5, (0, 255, 0), 2)        cv.imshow("1", frame)        # 等待30ms显示图像，若过程中按“Esc”退出        c = cv.waitKey(30) & 0xff        if c == 27:            capture.release()            breakvideo_demo()cv.destroyAllWindows()