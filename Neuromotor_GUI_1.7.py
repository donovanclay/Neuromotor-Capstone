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
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox, QMessageBox, QSizePolicy, QSpacerItem, QGridLayout
)
from PyQt5.QtGui import  QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import time
import sys
import pandas as pd
from xarray import DataArray
import os.path
from uuid import uuid4
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Processing_HUB coordinates communication between other classes
class Processing_HUB(QObject):
    Display1 = pyqtSignal(QImage)
    DeleteFrame=pyqtSignal()
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
                self.play_eeg_instance.Display1.connect(self.handle_Display1)
                self.play_eeg_instance.start()
        elif self.play_eeg_instance is not None and self.play_eeg_instance.isRunning():
            self.play_eeg_instance.requestInterruption()
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance = Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.Display1.connect(self.handle_Display1)
                self.play_eeg_instance.start()
        else:
            if self.eeg is not None and len(self.eeg) > 0:
                self.play_eeg_instance= Brain_video(self.eeg,self.spatial)
                self.play_eeg_instance.Display1.connect(self.handle_Display1)
                self.play_eeg_instance.start()

    def handle_Display1(self, qimage):
        self.Display1.emit(qimage)

    def update_indx(self, index): # not sure if this is needed but put it in anyways
         self.current_index=index

    @pyqtSlot()
    def stop_thread_play(self):
        if self.play_eeg_instance is not None:
            self.play_eeg_instance.pause_thread()
            self.pause=True 
            self.play_eeg_instance.requestInterruption()
            self.play_eeg_instance.deleteLater()
            self.play_eeg_instance.wait()
            self.play_eeg_instance=None
            self.DeleteFrame.emit()

# sends eeg imaging
class Brain_video(QThread):
    Display1 = pyqtSignal(QImage)
    def __init__(self,eeg_data,spatial_data, **kwargs):
        super(Brain_video, self).__init__()
        self.eeg_data=eeg_data
        self.spatial_data=spatial_data
        self.kwargs=kwargs
        self.ThreadActive=True
        self.vid_frame_rate = 10 # per second
        self.stop_thread = False
        self.play_index=0 # For get_video progress
        self.time_frame = 1 / self.vid_frame_rate
        self.average_half=round(0.5*self.vid_frame_rate)
        self.lag= 80 # miliseconds needed for sinking with movement but not used yet
        self.ch_type=None
        self.frame_index=self.vid_frame_rate

    def run(self):
        self.ThreadActive=True
        self.get_xy_coords()
        self.get_video()


    def get_video(self):
        self.stop_thread = False
        if self.eeg_data is not None:
            while self.ThreadActive:
                for i in range(self.frame_index, self.eeg_data[0].values.shape[0]-self.vid_frame_rate,self.vid_frame_rate):
                    if not self.stop_thread:
                        image_array = self.eeg_image(i)
                        if image_array is not None:
                            height, width, channel = image_array.shape
                            img_bytes = image_array.tobytes()
                            q_img = QImage(img_bytes, width, height, channel*width, QImage.Format_RGB888)
                            self.Display1.emit(q_img)
                            self.frame_index+=self.vid_frame_rate
                            QTimer.singleShot(int(self.time_frame * 1000), self.continue_video) 
                        else:
                            print('Failed to get frame')
                            continue
                    else:
                        print('Thread stopped')
                        break
            else:
                print('Failed to get EEG data')                

    def pause_thread(self):
        self.ThreadActive=False
        self.stop_thread=True

    def resume_thread(self):
        self.stop_thread=False
        self.ThreadActive=True

    def continue_video(self):
        if not self.stop_thread:
            self.get_video()  # Continue fetching frames

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
        self.current_index=0
        self.last_index=2
        self.can_go_back=[0,1,0] # 0 : cannot go back, 1: can go back
        self.show_vid1=[0,1,0]
        self.setWindowTitle("Insert title")
        self.setGeometry(80,80, 1000, 900)
        self.outerLayout = QVBoxLayout()
        self.topLayout = QGridLayout()
        self.Btm_layout = QVBoxLayout()
        self.MidLayout=QHBoxLayout()
        self.MidStack = QStackedWidget(self)
        self.init_new_widget()
        self.MidLayout.addWidget(self.MidStack)

 # Stacked Widget for switching between control sets

        self.prev_btn = QPushButton("Previous", self)
        self.prev_btn.setGeometry(20, 20, 200, 60)
        self.prev_btn.setVisible(False)
        self.prev_btn.clicked.connect(self.previous_control_set)
        self.topLayout.addWidget(self.prev_btn, 0, 0, Qt.AlignLeft)

        horizontalSpacer = QSpacerItem(40, 20,  QSizePolicy.Maximum, QSizePolicy.Expanding)
        self.topLayout.addItem(horizontalSpacer, 800, 0, Qt.AlignLeft)

                        # Current control set index and Next Button setup
        self.next_btn = QPushButton("Next", self)
        self.next_btn.setGeometry(600, 20, 200, 60)
        self.next_btn.clicked.connect(self.next_control_set)
        self.topLayout.addWidget(self.next_btn,0, 2, Qt.AlignRight )

        self.display_vid1 = QLabel(self)
        self.display_vid1.setFixedSize(640, 480)
        self.display_vid1.setGeometry(100, 100, 740, 580)
        # Allows for stacked layout which acts like tabs with separate pages
        self.current_layout = self.current_widget = ['Initial_data', 'CA_viz']  # names of each page
        self.topLayout.addWidget(self.display_vid1, 2,1, Qt.AlignLeft)

        self.outerLayout.addLayout(self.topLayout)
        self.outerLayout.addLayout(self.MidLayout)
        self.outerLayout.addLayout(self.Btm_layout)
        self.setLayout(self.outerLayout)
        self.thread= Processing_HUB()
        self.thread.Display1.connect(lambda image: self.displayFrame(image))
        self.thread.DeleteFrame.connect(self.delete_frame)
        # self.MidStack.setCurrentIndex(self.current_index)
        self.setWindowTitle("Welcome to eeg-based pridictor simulator")

    def init_new_widget(self):
            self.initial_data_loading()
            self.visual_denoising_verification()

    def next_control_set(self):
        if self.current_index < self.last_index:
            self.current_index += 1 
            self.MidStack.setCurrentIndex(self.current_index)
            if self.can_go_back[self.current_index]==1:
                 self.prev_btn.setVisible(True)
            if self.show_vid1[self.current_index]==1:
                 self.display_vid1.setVisible(True)
            self.send_preset_values()

        # changes Window based on area in pipeline
    def previous_control_set(self):
        if self.current_index > 0:
            self.stop_thread()
            self.current_index -= 1 
            if self.can_go_back[self.current_index]==0:
                self.prev_btn.setVisible(False)
            if self.show_vid1[self.current_index]==0:
                 self.display_vid1.setVisible(False)
            self.MidStack.setCurrentIndex(self.current_index)
            self.send_preset_values()

    def initial_data_loading(self):
            if self.current_index==0 and hasattr(MainWindow,'prev_btn'):
                self.prev_btn.setVisible(False)

            self.setWindowTitle("Selecting amount data")
            # Button label
            self.page_data_load=QWidget()
            # load data button (will be added to next button later)
            self.load_button = QPushButton(self.page_data_load, text="Select to load data")
            self.load_button.move(400, 80)
            self.load_button.setGeometry(400, 80, 200, 60)
            self.load_button.clicked.connect(self.get_initial_data)
            self. page1=self.MidStack.addWidget(self.page_data_load)
            print('intitializing widget 1st')
            return self. page1
        
    def visual_denoising_verification(self):
            self.page_CA_verification=QWidget()
            self.setWindowTitle("Component analysis verification")
            self.eeg_button = QPushButton("Select to play eeg", self.page_CA_verification)
            self.eeg_button.clicked.connect(self.play_eeg)
            self.eeg_button.move(450,20)
            self.eeg_button.setVisible(True)
            self.page2=self.MidStack.addWidget(self.page_CA_verification)
            print('intitializing widget 2nd')
            return self.page2

    @ pyqtSlot()
    def select_type_of_model(self): 
            self.thread.model()
            self.delete_widgets(self.select_model_button)
            if self.value==1:
                self.load_button.setVisible(True)

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
            self.display_vid1.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def get_initial_data(self): 
            self.thread.initial_file_opening()
            self.load_button.setVisible(False)

    @ pyqtSlot()
    def play_eeg(self): 
            self.thread.play_eeg()

    @pyqtSlot()
    def update_index(self):
            if 0 <= self.current_index < self.last_index:
                self.thread.update_indx(self.current_index)

    @ pyqtSlot()
    def stop_thread(self):
            self.thread.stop_thread_play()  

    @ pyqtSlot()
    def send_preset_values(self):
            print('send values')  

    @ pyqtSlot()
    def delete_frame(self):
            self.display_vid1.setVisible(False)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
