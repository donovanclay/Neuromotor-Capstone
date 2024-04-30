import sys
import time
import os
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect
import numpy as np
import functions_4_GUI as F4G
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
sys.setrecursionlimit(10**6)
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
# import cv2
import time
import sys
import pandas as pd
from xarray import DataArray
import os.path
from uuid import uuid4
import matplotlib

# Processing_HUB coordinates communication between other classes
class Processing_HUB(QObject):
    EEGFrame = pyqtSignal(QImage)
    
    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.eeg=None
        self.joint=None
        self.spatial=None
        self.play_eeg_instance=None

    def initial_file_opening(self, **kwargs):
        self.eeg=F4G.Eeg_data_function()
        self.joint=F4G.joint_data_function()
        self.spatial=F4G.spatial_data_function()

    def play_eeg(self):
        if self.play_eeg_instance is not None and self.play_eeg_instance.isFinished:
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance= Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()
        elif self.play_eeg_instance is not None and self.play_eeg_instance.isRunning():
            self.play_eeg_instance.requestInterruption()
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance = Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()
        else:
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance= Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()

    def handle_EEGFrame(self, qimage):
        self.EEGFrame.emit(qimage)

# sends eeg imaging
class Brain_video(QThread):
    EEGFrame = pyqtSignal(QImage)
    def __init__(self,eeg_data,spatial_data, **kwargs):
        super(Brain_video, self).__init__()
        self.eeg_data=eeg_data
        self.spatial_data=spatial_data
        self.kwargs=kwargs
        self.vid_frame_rate = 10 # per second
        self.stop_thread = False
        self.play_index=0 # For get_video progress
        self.time_frame = 1 / self.vid_frame_rate
        self.average_half=round(0.5*self.vid_frame_rate)
        self.lag= 80 # miliseconds needed for sinking with movement but not used yet
        self.ch_type=None

    def run(self):
        self.ThreadActive=True
        self.get_xy_coords()
        self.get_video()


    def get_video(self):
        self.stop_thread = False
        if self.eeg_data is not None:
            while self.ThreadActive:
                for i in range(self.vid_frame_rate, self.eeg_data[0].values.shape[0],self.vid_frame_rate):
                    if not self.stop_thread:
                        img = self.eeg_image(i)
                        if img is not None:
                            height, width,_ = img.shape
                            img = np.uint8(img) 
                            q_img = QImage(img, width, height, width, QImage.Format_BGR888)
                            self.EEGFrame.emit(q_img)
                            time.sleep(self.time_frame)
                        else:
                            print('Failed to get frame')
                            continue   
            #else:
            while self.stop_thread:
                        time.sleep(1*self.time_frame) 
            else:
                            print('Failed to get frame')                  

    def pause_thread(self):
        self.ThreadActive=False
        self.stop_thread=True

    def resume_thread(self):
        self.stop_thread=False
        self.ThreadActive=True

    import sys
import time
import os
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect
import numpy as np
import functions_4_GUI as F4G
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
sys.setrecursionlimit(10**6)
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
# import cv2
import time
import sys
import pandas as pd
from xarray import DataArray
import os.path
from uuid import uuid4
import matplotlib

# Processing_HUB coordinates communication between other classes
class Processing_HUB(QObject):
    EEGFrame = pyqtSignal(QImage)
    
    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.eeg=None
        self.joint=None
        self.spatial=None
        self.play_eeg_instance=None

    def initial_file_opening(self, **kwargs):
        self.eeg=F4G.Eeg_data_function()
        self.joint=F4G.joint_data_function()
        self.spatial=F4G.spatial_data_function()

    def play_eeg(self):
        if self.play_eeg_instance is not None and self.play_eeg_instance.isFinished:
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance= Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()
        elif self.play_eeg_instance is not None and self.play_eeg_instance.isRunning():
            self.play_eeg_instance.requestInterruption()
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance = Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()
        else:
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance= Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()

    def handle_EEGFrame(self, qimage):
        self.EEGFrame.emit(qimage)

# sends eeg imaging
class Brain_video(QThread):
    EEGFrame = pyqtSignal(QImage)
    def __init__(self,eeg_data,spatial_data, **kwargs):
        super(Brain_video, self).__init__()
        self.eeg_data=eeg_data
        self.spatial_data=spatial_data
        self.kwargs=kwargs
        self.vid_frame_rate = 10 # per second
        self.stop_thread = False
        self.play_index=0 # For get_video progress
        self.time_frame = 1 / self.vid_frame_rate
        self.average_half=round(0.5*self.vid_frame_rate)
        self.lag= 80 # miliseconds needed for sinking with movement but not used yet
        self.ch_type=None

    def run(self):
        self.ThreadActive=True
        self.get_xy_coords()
        self.get_video()


    def get_video(self):
        self.stop_thread = False
        if self.eeg_data is not None:
            while self.ThreadActive:
                for i in range(self.vid_frame_rate, self.eeg_data[0].values.shape[0],self.vid_frame_rate):
                    if not self.stop_thread:
                        img = self.eeg_image(i)
                        if img is not None:
                            height, width,_ = img.shape
                            img = np.uint8(img) 
                            q_img = QImage(img, width, height, width, QImage.Format_BGR888)
                            self.EEGFrame.emit(q_img)
                            time.sleep(self.time_frame)
                        else:
                            print('Failed to get frame')
                            continue   
            #else:
            while self.stop_thread:
                        time.sleep(1*self.time_frame) 
            else:
                            print('Failed to get frame')                  

    def pause_thread(self):
        self.ThreadActive=False
        self.stop_thread=True

    def resume_thread(self):
        self.stop_thread=False
        self.ThreadActive=True

    import sys
import time
import os
from PyQt5.QtCore import Qt, QObject, QRunnable, QThreadPool, QThread, pyqtSignal, pyqtSlot, QRect
import numpy as np
import functions_4_GUI as F4G
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
sys.setrecursionlimit(10**6)
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
# import cv2
import time
import sys
import pandas as pd
from xarray import DataArray
import os.path
from uuid import uuid4
import matplotlib

# Processing_HUB coordinates communication between other classes
class Processing_HUB(QObject):
    EEGFrame = pyqtSignal(QImage)
    
    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.eeg=None
        self.joint=None
        self.spatial=None
        self.play_eeg_instance=None

    def initial_file_opening(self, **kwargs):
        self.eeg=F4G.Eeg_data_function()
        self.joint=F4G.joint_data_function()
        self.spatial=F4G.spatial_data_function()

    def play_eeg(self):
        if self.play_eeg_instance is not None and self.play_eeg_instance.isFinished:
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance= Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()
        elif self.play_eeg_instance is not None and self.play_eeg_instance.isRunning():
            self.play_eeg_instance.requestInterruption()
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance = Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()
        else:
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance= Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.EEGFrame.connect(self.handle_EEGFrame)
                self.play_eeg_instance.start()

    def handle_EEGFrame(self, qimage):
        self.EEGFrame.emit(qimage)

# sends eeg imaging
class Brain_video(QThread):
    EEGFrame = pyqtSignal(QImage)
    def __init__(self,eeg_data,spatial_data, **kwargs):
        super(Brain_video, self).__init__()
        self.eeg_data=eeg_data
        self.spatial_data=spatial_data
        self.kwargs=kwargs
        self.vid_frame_rate = 10 # per second
        self.stop_thread = False
        self.play_index=0 # For get_video progress
        self.time_frame = 1 / self.vid_frame_rate
        self.average_half=round(0.5*self.vid_frame_rate)
        self.lag= 80 # miliseconds needed for sinking with movement but not used yet
        self.ch_type=None

    def run(self):
        self.ThreadActive=True
        self.get_xy_coords()
        self.get_video()


    def get_video(self):
        self.stop_thread = False
        if self.eeg_data is not None:
            while self.ThreadActive:
                for i in range(self.vid_frame_rate, self.eeg_data[0].values.shape[0],self.vid_frame_rate):
                    if not self.stop_thread:
                        image_array = self.eeg_image(i)
                        if image_array is not None:
                            height, width, channel = image_array.shape
                            img_bytes = image_array.tobytes()
                            q_img = QImage(img_bytes, width, height, channel*width, QImage.Format_RGB888)
                            self.EEGFrame.emit(q_img)
                            time.sleep(self.time_frame)
                        else:
                            print('Failed to get frame')
                            continue   
            #else:
            while self.stop_thread:
                        time.sleep(1*self.time_frame) 
            else:
                            print('Failed to get frame')                  

    def pause_thread(self):
        self.ThreadActive=False
        self.stop_thread=True

    def resume_thread(self):
        self.stop_thread=False
        self.ThreadActive=True

    def eeg_image(self, index_point):
        matplotlib.use('Agg')
        upper_bound = index_point + self.average_half
        lower_bound = index_point - self.average_half
        average_eeg = pd.DataFrame(self.eeg_data[0].isel(Recording=range(lower_bound, upper_bound)).values).mean()  # change 0 for more than 1 sess
        average_eeg = average_eeg.transpose()
        # Generate the topomap
        fig, ax = plt.subplots()
        plot_topomap(data=average_eeg, pos=self.xy, ch_type=self.ch_type, axes=ax)
        # Save the topomap as a pixel array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        image_array = np.frombuffer(buf, np.uint8).copy()
        image_array.shape = int(h), int(w), 4
        # Remove alpha channel
        image_array = image_array[:, :, :3]
        plt.close(fig)
        return image_array
      

    def get_xy_coords(self):
            x=self.spatial_data.sel(Values=1)[0].astype(float).to_numpy()
            y=self.spatial_data.sel(Values=2)[0].astype(float).to_numpy()
            x=x/(np.max(x))
            y=y/(np.max(y))
            self.xy=np.stack((x, y), axis=1)
            # print(self.xy.shape)
            if self.xy.shape!=self.eeg_data[0].values.shape[1]:
                self.eeg_data=self.eeg_data.drop_isel(Channel=-1)
                # print(self.eeg_data.shape)
            self.channels=len(self.spatial_data[0])
            # channel type from literature
            chan_type=['eeg']*self.channels
            EOG_indx=F4G.get_EoG_index()
            for EoG in EOG_indx:
                chan_type[EoG]='eog'
            self.ch_type=chan_type   

# sends joint imaging
class Movement_video(QThread):
    MoveFrame = pyqtSignal(QImage) 
    def __init__(self, **kwargs):
        super(Movement_video, self).__init__()

# joint angle predictions vs actual joint angles moving graph
class Accuracy_video(QThread):
    GraphFrame = pyqtSignal(QImage)
    def __init__(self, **kwargs):
        super(Accuracy_video, self).__init__()


# Applies Get_current_function to self.data_array and saves to Processing_HUB using
class Get_components(QThread):
        denoised=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, Numb_comp,type_CA,**kwargs):
            super(Get_components, self).__init__()
            self.eeg_data=eeg_data
            self.type=type_CA
            self.Numb_comp=Numb_comp

# Applies Transform
class Apply_tranform(QThread):  
        transform=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data,**kwargs):
            super(Apply_tranform, self).__init__()
            self.eeg_data=eeg_data

# Applies Machine learning algorythim
class Apply_Predictor(QThread):  
        Predictions=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, joint_data,**kwargs):
            super(Apply_Predictor, self).__init__()
            self.eeg=eeg_data
            self.joint=joint_data

# MainWindow handles all displays/interactions
class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("eeg_player")
        self.setGeometry(0, 0, 800, 500)
        outerLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        Button_layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.load_button = QPushButton("Select to load data", self)
        self.load_button.clicked.connect(self.get_initial_data)

        self.eeg_button = QPushButton("Select to play eeg", self)
        self.eeg_button.clicked.connect(self.play_eeg)
        self.eeg_button.setVisible(False)

        topLayout.addWidget(self.label)     
        topLayout.addWidget(self.load_button) #.deleteLater()
        topLayout.addWidget(self.eeg_button) #.deleteLater()
        topLayout.addWidget(self.eeg_button)

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(Button_layout)

        self.thread= Processing_HUB()
        self.thread.EEGFrame.connect(lambda image: self.displayFrame(image))

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def get_initial_data(self): 
        self.thread.initial_file_opening()
        self.load_button.deleteLater() 
        self.load_button=None
        self.eeg_button.setVisible(True)

    @ pyqtSlot()
    def play_eeg(self): 
        self.thread.play_eeg()
         
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    def get_xy_coords(self):
            x=self.spatial_data.sel(Values=1)[0].astype(float).to_numpy()
            y=self.spatial_data.sel(Values=2)[0].astype(float).to_numpy()
            x=x/(np.max(x))
            y=y/(np.max(y))
            self.xy=np.stack((x, y), axis=1)
            # print(self.xy.shape)
            if self.xy.shape!=self.eeg_data[0].values.shape[1]:
                self.eeg_data=self.eeg_data.drop_isel(Channel=-1)
                # print(self.eeg_data.shape)
            self.channels=len(self.spatial_data[0])
            # channel type from literature
            chan_type=['eeg']*self.channels
            EOG_indx=F4G.get_EoG_index()
            for EoG in EOG_indx:
                chan_type[EoG]='eog'
            self.ch_type=chan_type   

# sends joint imaging
class Movement_video(QThread):
    MoveFrame = pyqtSignal(QImage) 
    def __init__(self, **kwargs):
        super(Movement_video, self).__init__()

# joint angle predictions vs actual joint angles moving graph
class Accuracy_video(QThread):
    GraphFrame = pyqtSignal(QImage)
    def __init__(self, **kwargs):
        super(Accuracy_video, self).__init__()


# Applies Get_current_function to self.data_array and saves to Processing_HUB using
class Get_components(QThread):
        denoised=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, Numb_comp,type_CA,**kwargs):
            super(Get_components, self).__init__()
            self.eeg_data=eeg_data
            self.type=type_CA
            self.Numb_comp=Numb_comp

# Applies Transform
class Apply_tranform(QThread):  
        transform=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data,**kwargs):
            super(Apply_tranform, self).__init__()
            self.eeg_data=eeg_data

# Applies Machine learning algorythim
class Apply_Predictor(QThread):  
        Predictions=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, joint_data,**kwargs):
            super(Apply_Predictor, self).__init__()
            self.eeg=eeg_data
            self.joint=joint_data

# MainWindow handles all displays/interactions
class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("eeg_player")
        self.setGeometry(0, 0, 800, 500)
        outerLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        Button_layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.load_button = QPushButton("Select to load data", self)
        self.load_button.clicked.connect(self.get_initial_data)

        self.eeg_button = QPushButton("Select to play eeg", self)
        self.eeg_button.clicked.connect(self.play_eeg)
        self.eeg_button.setVisible(False)

        topLayout.addWidget(self.label)     
        topLayout.addWidget(self.load_button) #.deleteLater()
        topLayout.addWidget(self.eeg_button) #.deleteLater()
        topLayout.addWidget(self.eeg_button)

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(Button_layout)

        self.thread= Processing_HUB()
        self.thread.EEGFrame.connect(lambda image: self.displayFrame(image))

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def get_initial_data(self): 
        self.thread.initial_file_opening()
        self.load_button.deleteLater() 
        self.load_button=None
        self.eeg_button.setVisible(True)

    @ pyqtSlot()
    def play_eeg(self): 
        self.thread.play_eeg()
         
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

    def get_xy_coords(self):
            x=self.spatial_data.sel(Values=1)[0].astype(float).to_numpy()
            y=self.spatial_data.sel(Values=2)[0].astype(float).to_numpy()
            x=x/(np.max(x))
            y=y/(np.max(y))
            self.xy=np.stack((x, y), axis=1)
            # print(self.xy.shape)
            if self.xy.shape!=self.eeg_data[0].values.shape[1]:
                self.eeg_data=self.eeg_data.drop_isel(Channel=-1)
                # print(self.eeg_data.shape)
            self.channels=len(self.spatial_data[0])
            # channel type from literature
            chan_type=['eeg']*self.channels
            EOG_indx=F4G.get_EoG_index()
            for EoG in EOG_indx:
                chan_type[EoG]='eog'
            self.ch_type=chan_type   

# sends joint imaging
class Movement_video(QThread):
    MoveFrame = pyqtSignal(QImage) 
    def __init__(self, **kwargs):
        super(Movement_video, self).__init__()

# joint angle predictions vs actual joint angles moving graph
class Accuracy_video(QThread):
    GraphFrame = pyqtSignal(QImage)
    def __init__(self, **kwargs):
        super(Accuracy_video, self).__init__()


# Applies Get_current_function to self.data_array and saves to Processing_HUB using
class Get_components(QThread):
        denoised=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, Numb_comp,type_CA,**kwargs):
            super(Get_components, self).__init__()
            self.eeg_data=eeg_data
            self.type=type_CA
            self.Numb_comp=Numb_comp

# Applies Transform
class Apply_tranform(QThread):  
        transform=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data,**kwargs):
            super(Apply_tranform, self).__init__()
            self.eeg_data=eeg_data

# Applies Machine learning algorythim
class Apply_Predictor(QThread):  
        Predictions=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, joint_data,**kwargs):
            super(Apply_Predictor, self).__init__()
            self.eeg=eeg_data
            self.joint=joint_data

# MainWindow handles all displays/interactions
class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("eeg_player")
        self.setGeometry(0, 0, 800, 500)
        outerLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        Button_layout = QHBoxLayout()

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        self.load_button = QPushButton("Select to load data", self)
        self.load_button.clicked.connect(self.get_initial_data)

        self.eeg_button = QPushButton("Select to play eeg", self)
        self.eeg_button.clicked.connect(self.play_eeg)
        self.eeg_button.setVisible(False)

        topLayout.addWidget(self.label)     
        topLayout.addWidget(self.load_button) #.deleteLater()
        topLayout.addWidget(self.eeg_button) #.deleteLater()
        topLayout.addWidget(self.eeg_button)

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(Button_layout)

        self.thread= Processing_HUB()
        self.thread.EEGFrame.connect(lambda image: self.displayFrame(image))

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def get_initial_data(self): 
        self.thread.initial_file_opening()
        self.load_button.deleteLater() 
        self.load_button=None
        self.eeg_button.setVisible(True)

    @ pyqtSlot()
    def play_eeg(self): 
        self.thread.play_eeg()
         
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())