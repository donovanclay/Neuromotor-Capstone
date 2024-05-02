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
from PyQt5.QtGui import  QImage, QPixmap
from PyQt5.QtCore import QTimer
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

    def update_indx(self, index): # not sure if this is needed but put it in anyways
         self.current_index=index

    @pyqtSlot()
    def stop_thread_play(self):
        if self.play_eeg_instance is not None:
            self.play_eeg_instance.pause_thread()
            self.pause=True 
            self.play_eeg_instance.requestInterruption()
            self.play_eeg_instance.wait()
            self.play_eeg_instance=None
            self.DeleteFrame.emit()

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
                            self.EEGFrame.emit(q_img)
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
        self.Midlayo = None
        self.current_index=0
        self.last_index=1

        self.setWindowTitle("Insert title")
        self.setGeometry(80,80, 1000, 900)
        outerLayout = QVBoxLayout()
        topLayout = QHBoxLayout()
        Btm_layout = QVBoxLayout()
        MidLayout=QHBoxLayout()
        self.newLayout=None

        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)
        # Allows for stacked layout which acts like tabs with separate pages
        self.current_layout = self.current_widget = ['Initial_data', 'CA_viz']  # names of each page

        topLayout.addWidget(self.label)
        self.initial_data_loading()

        self.MidStack = QStackedWidget(self)
        MidLayout.addWidget(self.MidStack)

        self.layouts = []  # Store layouts here to prevent garbage collection

        for i in range(len(self.current_widget)):
            self.widget=QWidget()
            self.current_widget[i]=self.widget
            self.current_layout[i]=QVBoxLayout(self.current_widget[i])

        outerLayout.addLayout(topLayout)
        outerLayout.addLayout(MidLayout)

        if self.Midlayo is not None:
            outerLayout.addLayout(self.Midlayo)

        outerLayout.addLayout(Btm_layout)
        self.setLayout(outerLayout)

        self.thread= Processing_HUB()
        self.thread.EEGFrame.connect(lambda image: self.displayFrame(image))
        self.thread.DeleteFrame.connect(self.delete_frame)
        self.MidStack.setCurrentIndex(self.current_index)
        self.set_new_layout()
        self.setWindowTitle("Welcome to eeg-based pridictor simulator")

    # changes Window based on area in pipeline
    def next_control_set(self):
        self.remove_prev_widget()
        self.current_index += 1 
        self.update_index()
        self.set_new_layout()
        self.MidStack.setCurrentIndex(self.current_index)
        self.init_new_widget()
        self.send_preset_values()

    def previous_control_set(self):
        self.stop_thread()
        self.remove_prev_widget()
        self.current_index -= 1 
        self.update_index()
        self.set_new_layout()
        self.MidStack.setCurrentIndex(self.current_index)
        self.init_new_widget()
        self.send_preset_values()

    def remove_prev_widget(self):
        if self.Midlayo is not None:
            self.MidStack.removeWidget(self.current_widget[self.current_index])
            self.Midlayo.removeWidget(self.current_widget[self.current_index])

    def new_widget_handle(self, new_widget):
        if self.newLayout is not None:
            self.newLayout.addWidget(new_widget)
            self.MidStack.addWidget(self.widget)

    def set_new_layout(self):
        self.widget=QWidget()
        self.newLayout=QVBoxLayout(self.widget)

    def delete_widgets(self, widget):
        widget.deleteLater() 
        widget=None

    def init_new_widget(self):
        print('call function based on current_index')
        if self.current_index==0:
             self.initial_data_loading
        if self.current_index==1:
            self.visual_denoising_verification()
        elif self.current_index==2:
             print('Next page')
        elif self.current_index==self.last_index:
             print('You have reached the end of the simulator!')
        else:
             self.current_index=self.last_index

    def initial_data_loading(self):

        self.setWindowTitle("Selecting amount data")
        # Button label

        # load data button (will be added to next button later)
        self.load_button = QPushButton(self, text="Select to load data")
        self.load_button.move(400, 80)
        self.load_button.setGeometry(400, 80, 200, 60)
        self.new_widget_handle(self.load_button)  # .deleteLater()
        self.load_button.clicked.connect(self.get_initial_data)

        # Button to move to next screen
        self.next_btn = QPushButton("Next", self)
        self.new_widget_handle(self.next_btn)
        self.next_btn.move(900,20)
        self.next_btn.clicked.connect(self.next_control_set)

    def visual_denoising_verification(self):
        self.label.setVisible(True)
        self.setWindowTitle("Component analysis verification")
        self.eeg_button = QPushButton("Select to play eeg", self)
        self.eeg_button.clicked.connect(self.play_eeg)
        self.eeg_button.move(450,20)
        self.eeg_button.setVisible(True)
        self.new_widget_handle(self.eeg_button)

        self.prev_btn = QPushButton("Previous: Select components", self)
        self.prev_btn.clicked.connect(self.previous_control_set)
        self.new_widget_handle(self.prev_btn)

        self.next_btn = QPushButton("Next: Transforms", self)
        self.next_btn.clicked.connect(self.next_control_set)
        self.new_widget_handle(self.next_btn)

    @ pyqtSlot()
    def select_type_of_model(self): 
        self.thread.model()
        self.delete_widgets(self.select_model_button)
        if self.value==1:
            self.load_button.setVisible(True)

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def get_initial_data(self): 
        self.thread.initial_file_opening()
        self.delete_widgets(self.load_button) 

    @ pyqtSlot()
    def play_eeg(self): 
        self.thread.play_eeg()
        self.eeg_button.setVisible(False)

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
        self.label.setVisible(False)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
