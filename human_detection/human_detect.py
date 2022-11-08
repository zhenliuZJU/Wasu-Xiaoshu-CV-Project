import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 cutout, letterbox, mixup, random_perspective)

import numpy as np

class human_detection():
    def __init__(self):
        weights = './human_detection/runs/train/exp28/weights/best.pt'
        # weights = './runs/train/exp28/weights/best.pt'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = DetectMultiBackend(weights, device=device, dnn=False, data='./human_detection/data/people.yaml', fp16=False)
        # self.model = DetectMultiBackend(weights, device=device, dnn=False, data='./data/people.yaml',
                                        fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)
        self.img_size = 640
        self.auto = self.pt
        self.conf_thres = 0.3
        self.iou_thres = 0.35
        self.classes = None
        self.agnostic_nms = True
        self.max_det = 1000

    def human_detect(self, img_numpy):
        bs=1
        self.model.warmup(imgsz=(1 if self.pt or self. model.triton else bs, 3, *self.imgsz))  # warmup

        im = letterbox(img_numpy, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=None, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        det = pred[0]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_numpy.shape).round()

        return det
