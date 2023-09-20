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
from ikomia.utils import pyqtutils, qtconversion
from infer_yolor.infer_yolor_process import YoloRParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class YoloRWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = YoloRParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Dataset
        self.combo_dataset = pyqtutils.append_combo(self.grid_layout, "Trained on")
        self.combo_dataset.addItem("COCO")
        self.combo_dataset.addItem("Custom")
        self.combo_dataset.setCurrentIndex(0 if self.parameters.dataset == "COCO" else 1)
        self.combo_dataset.currentIndexChanged.connect(self.on_combo_dataset_changed)

    
        # Model model_path
        self.label_model_path = QLabel("Model path")
        self.browse_model = pyqtutils.BrowseFileWidget(path=self.parameters.model_weight_file, tooltip="Select file",
                                                       mode=QFileDialog.ExistingFile)
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_model_path, row, 0)
        self.grid_layout.addWidget(self.browse_model, row, 1)
        self.label_model_path.setVisible(False if self.parameters.dataset == "COCO" else True)
        self.browse_model.setVisible(False if self.parameters.dataset == "COCO" else True)

        # Model cfg
        self.label_config = QLabel("Config file")
        self.browse_config = pyqtutils.BrowseFileWidget(path=self.parameters.config_file, tooltip="Select file",
                                                     mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.label_config.setVisible(False if self.parameters.dataset == "COCO" else True)
        self.browse_config.setVisible(False if self.parameters.dataset == "COCO" else True)
        self.grid_layout.addWidget(self.label_config, row, 0)
        self.grid_layout.addWidget(self.browse_config, row, 1)

        # Input size
        self.spin_size = pyqtutils.append_spin(self.grid_layout, "Input size", self.parameters.input_size, step=2)

        # Confidence threshold
        self.spin_confidence = pyqtutils.append_double_spin(self.grid_layout, "Confidence", self.parameters.conf_thres,
                                                        step=0.05, decimals=2)

        # Iou threshold
        self.spin_iou = pyqtutils.append_double_spin(self.grid_layout, "IOU threshold", self.parameters.iou_thres,
                                                 step=0.05, decimals=2)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_combo_dataset_changed(self, index):
        if self.combo_dataset.itemText(index) == "COCO":
            self.label_model_path.setVisible(False)
            self.browse_model.setVisible(False)
            self.browse_model.set_path(self.combo_model.currentText() + ".pt")
            self.combo_model.setVisible(True)
            self.browse_config.setVisible(False)
            self.label_config.setVisible(False)
        else:
            self.label_model_path.setVisible(True)
            self.browse_model.setVisible(True)
            self.combo_model.setVisible(False)
            self.browse_config.setVisible(True)
            self.label_config.setVisible(True)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.dataset = self.combo_dataset.currentText()
        self.parameters.model_weight_file = self.browse_model.path
        self.parameters.input_size = self.spin_size.value()
        self.parameters.conf_thres = self.spin_confidence.value()
        self.parameters.iou_thres = self.spin_iou.value()
        self.parameters.config_file = self.browse_config.path

        # update model
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class YoloRWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_yolor"

    def create(self, param):
        # Create widget object
        return YoloRWidget(param, None)
