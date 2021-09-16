from ikomia.dnn import dataset as ikdataset

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from YoloRTrain.yolor.utils.torch_utils import select_device
from YoloRTrain.yolor.models.models import *
from YoloRTrain.yolor.utils.plots import plot_one_box


def load_image(path_img,img_size,auto_size):
    img0 = cv2.imread(path_img)  # BGR
    assert img0 is not None, 'Image Not Found ' + path_img
    img_size = img_size//auto_size*auto_size
    # Padded resize
    img = cv2.resize(img0, (img_size,img_size), interpolation = cv2.INTER_AREA)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return path_img, img, img0

def infere(model,img,im0,device,names,conf_thres,iou_thres):
    half = device.type != 'cpu'  # half precision only supported on CUDA
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    if half:
        model.half()  # to FP16
    img = torch.from_numpy(img).to(device, non_blocking=True)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
    return im0

imgsz = 512
gs = 64
path_img,img,im0 = load_image("/home/ambroise/Developpement/wgisd/data/CDY_2015.jpg",imgsz,gs)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = select_device(device)
cfg = "cfg/yolor_test.cfg"
weights = "tenta1/weights/best.pt"
conf_thres = 0.3
iou_thres = 0.6
classes = None
agnostic_nms = False
names=['raisin']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# Load model
model = Darknet(cfg).cuda()
model.load_state_dict(torch.load(weights, map_location=device))

model.to(device).eval()

with torch.no_grad():
    res = infere(model,img,im0,device,names,conf_thres,iou_thres)

plt.imshow(im0)
plt.savefig("super_inference.png")