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

from ikomia import utils, core, dataprocess
import copy
import torch
from pathlib import Path
import os
#from infer_yolor.yolor.utils.google_utils import gdrive_download
from infer_yolor.yolor.models.models import *
from infer_yolor.yolor.utils.torch_utils import select_device
import numpy as np
from torchvision.transforms import Resize
from infer_yolor.yolor.utils.general import non_max_suppression, scale_coords
import requests


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class YoloRParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.config_file = ""
        self.model_weight_file = ""
        self.dataset = "COCO"
        self.input_size = 512
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.cuda = torch.cuda.is_available()

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.input_size = int(params["input_size"])
        self.config_file = str(params["config_file"])
        self.model_weight_file = str(params["model_weight_file"])
        self.dataset = str(params["dataset"])
        self.conf_thres = float(params["conf_thres"])
        self.iou_thres = float(params["iou_thresh"])
        self.agnostic_nms = utils.strtobool(params["agnostic_nms"])
        self.cuda = utils.strtobool(params["cuda"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
            "input_size": str(self.input_size), 
            "config_file": self.config_file, 
            "model_weight_file": self.model_weight_file,
            "dataset": self.dataset,
            "conf_thres": str(self.conf_thres),
            "iou_thresh": str(self.iou_thres),
            "agnostic_nms": str(self.agnostic_nms),
            "cuda": str(self.cuda)
            }
        return params


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class YoloRProcess(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        self.model = None
        self.update = False
        self.config_file = None
        self.model_weight_file = ""
        self.model_path = None
        # Detect if we have a GPU available
        self.device = torch.device("cpu")

        # Create parameters class
        if param is None:
            self.set_param_object(YoloRParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.begin_task_run()

        # Temporary fix to clean detection outputs
        self.get_output(1).clear_data()

        # Get parameters :
        param = self.get_param_object()

        # Display all classes
        classes = None

        if param.model_weight_file != "":
                param.dataset ="Custom"

        if param.dataset == "COCO":
            # Get weight_path
            self.model_path = Path(
                os.path.dirname(os.path.realpath(__file__)), "yolor", "models", "yolor_p6.pt")

            if not os.path.isfile(self.model_path):
                model_url = utils.get_model_hub_url() + "/" + self.name + "/yolor_p6.pt"
                print("Downloading weights...")
                response = requests.get(model_url, stream=True)
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Weights downloaded")

            self.config_file = Path(os.path.dirname(os.path.realpath(__file__)) + "/yolor/cfg/" + "yolor_p6.cfg")
            name_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "yolor", "data", "coco.names")
            self.read_class_names(name_file_path)
            ckpt = torch.load(self.model_path)

        if param.dataset == "Custom":
            self.config_file = param.config_file
            self.model_path = param.model_weight_file
            ckpt = torch.load(self.model_path)
            if 'names' in ckpt.keys():
                self.set_names(ckpt['names'])

        if self.model is None or param.update:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.model = Darknet(self.config_file.__str__()).to(self.device)
            self.model.eval()
            # state_dict = {k: v for k, v in ckpt['model'].items() if self.model.state_dict()[k].numel() == v.numel()}
            state_dict = ckpt['model']
            self.model.load_state_dict(state_dict, strict=True)
            print('Transferred %g/%g items from %s' %
                  (len(state_dict), len(self.model.state_dict()), self.model_path))  # report
            param.update = False

        self.emit_step_progress()

        if self.model is not None:
            img_input = self.get_input(0)
            src_image = img_input.get_image()
            self.model.to(self.device)
            with torch.no_grad():
                self.detect(src_image, param.input_size, param.conf_thres, param.iou_thres, classes, param.agnostic_nms)

        # Call endTaskRun to finalize process
        self.emit_step_progress()
        self.end_task_run()

    def detect(self, im0, imgsz, conf_thres, iou_thres, classes, agnostic_nms):
        half = False  # for this model half precision does not work in pytorch 1.9
        if half:
            self.model.half()  # to FP16

        # Run inference
        h, w, _ = np.shape(im0)
        img = np.ascontiguousarray(im0)

        img = torch.from_numpy(img)
        img = img.to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img = img.permute(0, 3, 1, 2)
        img = Resize((imgsz, imgsz))(img)

        inf_out = self.model(img)[0].to('cpu')
        output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes,
                                     agnostic=agnostic_nms)[0]

        # Rescale boxes from img_size to im0 size
        whwh = torch.tensor([w / imgsz, h / imgsz, w / imgsz, h / imgsz]).to('cpu')
        for pred in output:
            pred[:4] *= whwh

        index = 0
        for *xyxy, conf, cls in output:
            # Box
            w = float(xyxy[2] - xyxy[0])
            h = float(xyxy[3] - xyxy[1])
            self.add_object(index, int(cls), conf.item(), float(xyxy[0]), float(xyxy[1]), w, h)
            index += 1


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class YoloRProcessFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolor"
        self.info.short_description = "Inference for YoloR object detection models"
        self.info.authors = "Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.6"
        self.info.icon_path = "icons/icon.png"
        self.info.article = "You Only Learn One Representation: Unified Network for Multiple Tasks"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "GPL-3.0 License"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2105.04206"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_yolor"
        self.info.original_repository = "https://github.com/WongKinYiu/yolor"
        # Keywords used for search
        self.info.keywords = "yolo, inference, pytorch, object, detection"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create process object
        return YoloRProcess(self.info.name, param)
