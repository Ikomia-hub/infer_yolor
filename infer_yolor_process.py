# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
import torch
from pathlib import Path
import os

from infer_yolor.yolor.utils.google_utils import gdrive_download
from infer_yolor.yolor.models.models import *
from infer_yolor.yolor.utils.torch_utils import select_device
import random
import numpy as np
from torchvision.transforms import Resize
from infer_yolor.yolor.utils.general import non_max_suppression, scale_coords


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class YoloRParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.cfg = ""
        self.weights = ""
        self.dataset = "COCO"
        self.input_size = 512
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.model_name = "yolor_p6"

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.input_size = int(param_map["input_size"])
        self.cfg = str(param_map["cfg"])
        self.weights = str(param_map["weights"])
        self.dataset = str(param_map["dataset"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thresh"])
        self.agnostic_nms = bool(param_map["agnostic_nms"])
        self.model_name = str(param_map["model_name"])
        self.update = bool(param_map["update"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class YoloRProcess(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.model = None
        self.names = None
        self.colors = None
        self.update = False
        self.cfg = None
        # Detect if we have a GPU available
        self.device = select_device("cuda" if torch.cuda.is_available() else "cpu")
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CBlobMeasureIO())

        # Create parameters class
        if param is None:
            self.setParam(YoloRParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Display all classes
        classes = None

        if param.dataset == "COCO":
            # Get weight_path
            self.weights = Path(os.path.dirname(os.path.realpath(__file__))+"/yolor/models/"+param.model_name+".pt")
            pretrained_models = {'yolor_p6': '1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76',
                                 'yolor_w6': '1UflcHlN5ERPdhahMivQYCbWWw7d2wY7U'}

            if not(self.weights.exists()):
                gdrive_download(id = pretrained_models[param.model_name], name=self.weights.__str__())

            self.cfg = Path(os.path.dirname(os.path.realpath(__file__))+"/yolor/cfg/"+param.model_name+".cfg")

            with open(Path(os.path.dirname(os.path.realpath(__file__))+"/yolor/data/coco.names")) as f:
                self.names = f.read().split("\n")[:-1]
            ckpt = torch.load(self.weights)

        if param.dataset == "Custom":
            self.cfg = param.cfg
            self.weights = param.weights
            ckpt = torch.load(self.weights)
            if 'names' in ckpt.keys():
                self.names = ckpt['names']

        if self.model is None or param.update:
            self.model = Darknet(self.cfg.__str__()).to(self.device)
            self.model.eval()
            # state_dict = {k: v for k, v in ckpt['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
            state_dict = ckpt['model']
            self.model.load_state_dict(state_dict, strict=True)
            print('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), self.weights))  # report
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            param.update = False

        if self.model is not None:
            img_input = self.getInput(0)
            src_image = img_input.getImage()
            graphics_output = self.getOutput(1)
            graphics_output.setNewLayer("YoloR")
            graphics_output.setImageIndex(0)
            # Init numeric output
            numeric_output = self.getOutput(2)
            numeric_output.clearData()

            # Forward input image
            self.forwardInputImage(0, 0)
            with torch.no_grad():
                self.detect(self.model, src_image, self.names, self.device, param.input_size, param.conf_thres,
                            param.iou_thres, classes, param.agnostic_nms, graphics_output, numeric_output)

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def detect(self, model, im0, names, device, imgsz, conf_thres, iou_thres, classes, agnostic_nms, graphics_output,
               numeric_output):

        half = device.type != 'cpu'  # half precision only supported on CUDA
        model.to(device).eval()

        if half:
            model.half()  # to FP16

        # Run inference
        h, w, _ = np.shape(im0)
        img = np.ascontiguousarray(im0)

        img = torch.from_numpy(img)
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img=img.permute(0, 3, 1, 2)
        img = Resize((imgsz, imgsz))(img)

        inf_out = model(img)[0]
        output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes,
                                     agnostic=agnostic_nms)[0]
        # Rescale boxes from img_size to im0 size
        whwh = torch.tensor([w / imgsz, h / imgsz, w / imgsz, h / imgsz]).to(device)
        for pred in output:
            pred[:4] *= whwh

        for *xyxy, conf, cls in output:
            # Box
            w = float(xyxy[2] - xyxy[0])
            h = float(xyxy[3] - xyxy[1])
            prop_rect = core.GraphicsRectProperty()
            prop_rect.pen_color = self.colors[int(cls)]
            graphics_box = graphics_output.addRectangle(float(xyxy[0]), float(xyxy[1]), w, h, prop_rect)
            graphics_box.setCategory(self.names[int(cls)])
            # Label
            name = names[int(cls)]
            prop_text = core.GraphicsTextProperty()
            prop_text.font_size = 8
            prop_text.color = self.colors[int(cls)]
            prop_text.bold = True
            graphics_output.addText(name, float(xyxy[0]), float(xyxy[1]), prop_text)
            # object results
            results = []
            confidence_data = dataprocess.CObjectMeasure(dataprocess.CMeasure(core.MeasureId.CUSTOM, "Confidence"),
                                                         conf.item(),
                                                         graphics_box.getId(),
                                                         name)
            box_data = dataprocess.CObjectMeasure(dataprocess.CMeasure(core.MeasureId.BBOX),
                                                  [float(xyxy[0]), float(xyxy[1]), w, h],
                                                  graphics_box.getId(),
                                                  name)
            results.append(confidence_data)
            results.append(box_data)
            numeric_output.addObjectMeasures(results)


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class YoloRProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolor"
        self.info.shortDescription = "Inference for YoloR object detection models"
        self.info.description = "Inference for YoloR object detection models." \
                                "You Only Learn One Representation: Unified Network for Multiple Tasks"
        self.info.authors = "Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.iconPath = "icons/icon.png"
        self.info.article = "You Only Learn One Representation: Unified Network for Multiple Tasks"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "GPL-3.0 License"
        # URL of documentation
        self.info.documentationLink = "https://arxiv.org/abs/2105.04206"
        # Code source repository
        self.info.repository = "https://github.com/WongKinYiu/yolor"
        # Keywords used for search
        self.info.keywords = "yolo, inference, pytorch, object, detection"

    def create(self, param=None):
        # Create process object
        return YoloRProcess(self.info.name, param)
