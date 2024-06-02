import sys
import time
import os
from PyQt5.QtCore import Qt, QObject,  Q_ARG, QMetaObject, QThread, pyqtSignal, pyqtSlot, QRect
import numpy as np
import functions_4_GUI6 as F4G
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
sys.setrecursionlimit(10**6)
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QProgressBar,
    QPushButton, QFileDialog, QWidget, QSlider, QStackedWidget, QComboBox, QMessageBox, QSizePolicy, QSpacerItem, QGridLayout
)
from PyQt5.QtGui import  QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import matplotlib.pyplot as plt
import time
import sys
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
import Functions_4_model as FM
import os.path
from sklearn.decomposition import FastICA
from uuid import uuid4
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import scipy.signal as sig
import matplotlib.patches as patches
import mne

# Processing_HUB coordinates communication between other classes
class Processing_HUB(QObject):
    Display1 = pyqtSignal(QImage)
    Display2 = pyqtSignal(QImage)

    def __init__(self):
        super(Processing_HUB, self).__init__()
        self.eeg=None
        self.joint=None
        self.velocity=None
        self.spatial=None
        self.play_eeg_instance=None
        self.subject_dict={'/SL01':1,'/SL02':2,'/SL03':3,'/SL04':4, '/SL05':5,'/SL06':6,'/SL07':7,'/SL08':8}
        self.data_size_dictionary= {'Single subject: One session':0,'Single subject: Three sessions':1 , 'Entire dataset':2}
        self.training_dataset=None
        self.test_dataset=None
        self.data_selected=self.data_size_dictionary['Single subject: One session']
        self.eeg_raw=None
        self.slider_value=None
        self.slider_value_3=None
        self.slider_value_2=None
        self.slider_value_4=None
        self.current_index=0
        self.fs=100 # frequency of eeg
        self.decimal_tenths=[0] # indices where decimals must be used
        self.decimal_hundreths_2=[1]
        self.parameters=None
        self.subject_selected=0
        self.function_methods=[3] # index where methods are passed
        self.method=None
        self.ica_comp=30
        self.play_eeg_instance2=None
        self.Update_previous=[2,3] # index where previous eeg is updated
        self.velocity_array=None
        self.joint_array=None
        self.first_graph=0 # variable to aviod calling function multiple times
        self.first_graph1=0
        self.first_graph2=0
        self.method= None
        # ***
        self.Length_array=None
        self.graph_intvl=[100, 300]
        self.off_set= 5 # (0.05 seconds)
        self.channel1= 0
        self.channel2= 3
        self.page_offset =1
        self.velocity_array=None
        self.Length_array=None

### Functions below are for data loading
    # *** New
    def eeg_folder_path(self, folder_path):
        eeg_data=xr.load_dataset(folder_path)
        eeg_=eeg_data.isel(Trial=0).copy()
        eeg=eeg_.__xarray_dataarray_variable__
        self.eeg_array=np.array(eeg.values)
        self.Length_array=range(len(self.eeg_array))
    
    def velocity_folder_path(self,folder_path):
        velocity_data=xr.load_dataset(folder_path)
        velocity_=velocity_data.isel(Trial=0).copy()
        velocity=velocity_.__xarray_dataarray_variable__
        self.velocity_array=np.array(velocity.values)
        self.initial_data_processing()

    def initial_data_processing(self):
         self.offset()
         self.velocity_normalizer()

    def offset(self):
         self.eeg_array,self.velocity_array=FM.align_data_by_offset(self.eeg_array,self.velocity_array,self.off_set)

    def velocity_normalizer(self):
        self.velocity_array=FM.lp(self.velocity_array,30)
        ica = FastICA(n_components=1) 
        for i in range(self.velocity_array.shape[1]):
            self.velocity_array[:, i] = ica.fit_transform(self.velocity_array[:, i].reshape(-1, 1)).flatten()
        self.velocity_array,scalerV =FM.velocity_rescaler(self.velocity_array)
        for i in range(self.velocity_array.shape[1]):
            self.velocity_array[:, i] = FM.filter_and_interpolate(self.velocity_array[:, i],1)
         
# ***


    def function_controler(self):
            if self.current_index==self.page_offset:
                # *** 
                self.graph_velocity()

    def send_image_Display1(self, image_array):
         if image_array is not None:
                height, width, channel = image_array.shape
                img_bytes = image_array.tobytes()
                q_img = QImage(img_bytes, width, height, channel*width, QImage.Format_RGB888)
                self.handle_Display1(q_img)
    
    def send_image_Display2(self, image_array):
         if image_array is not None:
                height, width, channel = image_array.shape
                img_bytes = image_array.tobytes()
                q_img = QImage(img_bytes, width, height, channel*width, QImage.Format_RGB888)
                self.handle_Display2(q_img)
     # ***       
    def graph_velocity(self):
        matplotlib.use('Agg')
        fig, ax = plt.subplots()
        
        # Determine the middle point and calculate the shading region
        middle_x = (self.graph_intvl[1]-self.graph_intvl[0]) / 2
        left_x = middle_x - self.slider_value / 2
        right_x = middle_x + self.slider_value / 2

           # Calculate the interval for the average calculation
        left_idx = int()
        right_idx = int(min(self.graph_intvl[1], right_x))
        
        # Plot the velocity signal
        if self.current_index == self.page_offset:
            plt.plot(self.velocity_array[self.graph_intvl[0]:self.graph_intvl[1], self.channel1],
                    label='Channel 1', alpha=0.7)
        if self.current_index == self.page_offset:
            plt.plot(self.velocity_array[self.graph_intvl[0]:self.graph_intvl[1], self.channel2],
                    label='Channel 4', alpha=0.7)
            
        # Calculate and plot the average line for Channel 1
        avg_channel1 = np.mean(self.velocity_array[left_idx:right_idx, self.channel1])
        ax.axhline(y=avg_channel1, xmin=(((left_x)/(2*middle_x  ))) , 
                   xmax=((right_x)/(2*middle_x  )) , color='green', linestyle='--', label=f'Avg Channel 1: {avg_channel1:.2f}')
        
        # Calculate and plot the average line for Channel 2
        avg_channel2 = np.mean(self.velocity_array[left_idx:right_idx, self.channel2])
        ax.axhline(y=avg_channel2, xmin=(left_x/(2*middle_x  )),
                   xmax=(right_x/(2*middle_x  )) , color='red', linestyle='--', label=f'Avg Channel 4: {avg_channel2:.2f}')
        ax.axvspan(left_x, right_x, color='blue', alpha=0.3)
        plt.title('Velocity Signal')
        plt.xlabel('Time (ms)')
        plt.ylabel('Normalized Velocity')
        plt.legend(loc='upper right')
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        image_array = np.frombuffer(buf, np.uint8).copy()
        image_array.shape = int(h), int(w), 4
        image_array = image_array[:, :, :3]
        plt.close(fig)
        self.send_image_Display1(image_array)

    def graph_components(self):
        matplotlib.use('Agg')
        fig, ax = plt.subplots()
        plt.plot(range(len(self.cum_exp_var)), self.cum_exp_var, label='Varience explained by components {} %'.format(round(self.cum_exp_var[self.slider_value-1])), alpha=0.7)
        plt.plot(self.slider_value,self.cum_exp_var[self.slider_value-1],'.', 'MarkerSize',10, color='orange')
        plt.title('Varience:{}'.format(self.method))
        plt.xlabel('Number of components {} '.format(self.slider_value))
        plt.ylabel('Cumulative variance')
        plt.legend()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        l, b, w, h = fig.bbox.bounds
        image_array = np.frombuffer(buf, np.uint8).copy()
        image_array.shape = int(h), int(w), 4
        # Remove alpha channel
        image_array = image_array[:, :, :3]
        plt.close(fig)
        if image_array is not None:
                height, width, channel = image_array.shape
                img_bytes = image_array.tobytes()
                q_img = QImage(img_bytes, width, height, channel*width, QImage.Format_RGB888)
        self.handle_Display2(q_img)  

    def handle_Display1(self, qimage):
        self.Display1.emit(qimage)

    def handle_Display2(self, qimage):
        self.Display2.emit(qimage)

    def get_preset_values(self, current_index, param):
        if current_index in self.Update_previous:
            self.data_for_viz()
        self.current_index = current_index
        if param is not None:
            vectorized_is_string = np.vectorize(self.is_non_digit_string)
            string_indices = np.where(vectorized_is_string(param))[0]
            if string_indices.size > 0:
                index = string_indices[0]
                self.method = param[index]
                remaining_array = np.delete(param, index)
                param = remaining_array
                param = [int(x) if isinstance(x, str) and x.isdigit() else x for x in param]
            if len(param) > 0:
                self.slider_value = param[0]
                if len(param) > 1:
                    self.slider_value_2 = param[1]
                    if len(param) > 2:
                        self.slider_value_3 = param[2]
                self.parameters_adjust()
        if self.current_index == 2:
            self.eeg_previous = self.eeg_previous.to_numpy()
            self.pca = PCA(n_components=60, random_state=1)
            self.pca.fit(self.eeg_previous)
            # Calculate cumulative explained variance across all PCs
            self.cum_exp_var = self.variance()
        self.function_controler()

    def is_non_digit_string(self, x):
        return isinstance(x, str) and not x.isdigit()
    
    def parameters_adjust(self):
        if self.eeg_array is not None and len(self.eeg_array) > 0:
            if self.current_index in self.decimal_tenths:
                self.slider_value= (self.slider_value/10)
            if self.current_index in self.decimal_hundreths_2:
                self.slider_value_2 = (self.slider_value_2 / 100)

    def update_indx(self, index): 
         self.current_index=index

    def change_method(self, method):
        self.method=method 
        self.function_controler()
    
    def temp_mod_1(self, value): # takes value from on__change and adjusts value for functions
        self.slider_value = value
        if self.current_index in self.decimal_tenths:
            self.slider_value=value/10
        self.function_controler()
        
    def temp_mod_2(self, value_2):
        self.slider_value_2=value_2
        if self.current_index in self.decimal_hundreths_2:
            self.slider_value_2=self.slider_value_2/100
        self.function_controler()

    def temp_mod_3(self, value_3):
        self.slider_value_3=value_3
        self.function_controler()

    def temp_method(self,method):
         self.method=method
         self.function_controler()

class delayed_data_processing(QThread):
    joint_data_processed = pyqtSignal(object, object)
    def __init__(self, subject, trials):
        super().__init__()
        self.subject = subject
        self.trials = trials

    def run(self):
        joint_array, velocity_array = F4G.joint_data_function(self.subject, self.trials, [2, 17])
        self.joint_data_processed.emit(joint_array, velocity_array)       


# MainWindow handles all displays/interactions
class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.current_index=0
        self.last_index=4
        self.decimal_tenths=[0] # indices where decimals must be used
        self.decimal_hundreths_2=[1]
        self.opersys=0 # default value
        self.method_indx= [2] # index for when methods are selectable
        self.init_method=['PCA']
        self.can_go_back=[0,0,0,1,0,0] # 0 : cannot go back, 1: can go back
        self.show_vid1=[1,1,1,1,0]
        self.show_vid2=[0,0,1,1,0,0]
        self.thread= Processing_HUB()
        # ***
        self.page_offset=1
        self.Min_slider =[[0],[10],[1],[0],[0],[0]] #  second value changed
        self.Max_slider = [[0],[50],[40],[0],[0],[0]] # 
        self.slider_name =[['None'], ['Window'], ['Number of Components'], ['None'], ['None']]
        # ***
        self.methods=[['PCA','ICA']]
        self.oper_type=['Windows:environment WSL2', 'Non-Windows: WSL2', 'Opsys not supported']
        self.select_subject={'Subject 1':1,'Subject 2':2, 'Subject 3':3,'Subject 4':4,
                                   'Subject 5':5 ,'Subject 6': 6,'Subject 7': 7,'Subject 8':8}
        self.data_size=['Single subject: One session','Single subject: Three sessions'] # 'Entire dataset' not yet included 
        # though it is supported in some functions
        self.setWindowTitle("Insert title")
        self.init_slider=[[0],[5,5,40],[10], [0],[0]]
        self.apply_perm_function=[2,4]
        self.setGeometry(80,80, 1000, 900)
        self.outerLayout = QVBoxLayout()
        self.topLayout = QGridLayout()
        self.Btm_layout = QGridLayout()
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
        horizontalSpacer2 = QSpacerItem(40, 20,  QSizePolicy.Maximum, QSizePolicy.Expanding)
        self.Btm_layout.addItem(horizontalSpacer2,800, 0, Qt.AlignLeft)
        # Current control set index and Next Button setup
        self.next_btn = QPushButton("Next", self)
        self.next_btn.setGeometry(600, 20, 200, 60)
        self.next_btn.clicked.connect(self.next_control_set)
        self.topLayout.addWidget(self.next_btn,0, 2, Qt.AlignRight )
        self.next_btn.setVisible(False)

        self.display_vid1 = QLabel(self)
        self.MidStack.setFixedSize(1200, 200)
        self.setFixedSize(1200, 900)
        self.display_vid1.setScaledContents(True) 
        self.topLayout.addWidget(self.display_vid1, 2,1, Qt.AlignLeft)
        self.display_vid2 = QLabel(self)
        self.display_vid2.setScaledContents(True) 
        self.empty_label = QLabel('',self)
        self.Btm_layout.addWidget(self.display_vid2,2,1, Qt.AlignLeft)

        self.outerLayout.addLayout(self.topLayout)
        self.outerLayout.addLayout(self.MidLayout)
        self.outerLayout.addLayout(self.Btm_layout)
        self.setLayout(self.outerLayout)
        self.thread.Display1.connect(lambda image: self.displayFrame(image))
        self.thread.Display2.connect(lambda image: self.displayFrame2(image))
        self.setWindowTitle("Welcome to eeg-based pridictor simulator")

    def init_new_widget(self):
            self.initial_data_loading()
            self.transform_page()
            self.Button_manager=[[0],[0],[0],[0],[0], [0]] # , self.Box_labelca

    def next_control_set(self):
        if self.current_index < self.last_index:
            self.preVisability_manager()
            self.current_index += 1 
            self.MidStack.setCurrentIndex(self.current_index)
            self.postVisability_manager()
            self.send_preset_values()

        # changes Window based on area in pipeline
    def previous_control_set(self):
        if self.current_index > 0:
            self.stop_thread()
            self.preVisability_manager()
            self.current_index -= 1 
            self.postVisability_manager()
            self.MidStack.setCurrentIndex(self.current_index)
            self.send_preset_values()

    def postVisability_manager(self):
        if self.can_go_back[self.current_index]==1:
                self.prev_btn.setVisible(True)
        elif self.can_go_back[self.current_index]==0:
             self.prev_btn.setVisible(False)
        if self.show_vid1[self.current_index]==1:
                 self.display_vid1.setVisible(True)
        if self.show_vid2[self.current_index]==1:
                 self.display_vid2.setVisible(True)
        if self.show_vid1[self.current_index]==0:
                 self.display_vid1.setVisible(False)
        if self.show_vid2[self.current_index]==0:
                 self.display_vid2.setVisible(False)
        if self.Button_manager[self.current_index][0]!=0:
             for button in self.Button_manager[self.current_index]:
                  button.setVisible(True)

    def preVisability_manager(self):
        if self.Button_manager[self.current_index][0]!=0:
             for button in self.Button_manager[self.current_index]:
                  button.setVisible(False)
         

    def initial_data_loading(self):
            # sets page 1 widget
            self.setWindowTitle("Selecting amount data")
            self.page_data_load=QWidget()

            # Selecting os
            self.Opselect = QComboBox(self.page_data_load)
            self.Opselect.setGeometry(700, 30, 200, 60)
            self.Opselect.addItems(self.oper_type) 
            self.Ops_label = QLabel('Select os & env', self.page_data_load)
            self.Ops_label.move(700, 10)
            self.Opselect.currentIndexChanged.connect(self.on_os_changed)

            self.confirm_button = QPushButton(self.page_data_load, text="Press to confirm")
            self.confirm_button.setGeometry(920, 30, 200, 60)
            self.confirm_button.clicked.connect(self.path_selector)
            # ***
            self.eeg_button = QPushButton(self.page_data_load, text="Load filtered eeg data")
            self.eeg_button.setGeometry(300, 10, 200, 60)
            self.eeg_button.setVisible(False)
            self.eeg_button.clicked.connect(self.eeg_folder)
            # ***
            # ***
            self.velocity_button = QPushButton(self.page_data_load, text="Load velocity data")
            self.velocity_button.setGeometry(600, 10, 200, 60)
            self.velocity_button.clicked.connect(self.velocity_folder)
            self.velocity_button.setVisible(False)
            # ***
            self. page1=self.MidStack.addWidget(self.page_data_load)
            return self. page1
    # ***
    def transform_page(self):
        self.page_trans=QWidget()
        self.slider_deck(self.page_trans,self.page_offset)
        self.page3=self.MidStack.addWidget(self.page_trans)
        return self.page3
    
    # ***   
    def on_os_changed(self):
        self.opersys=self.Opselect.currentIndex()

    def path_selector(self):
        self.eeg_button.setVisible(True)
        self.Opselect.setVisible(False)
        self.Ops_label.setVisible(False)
        self.confirm_button.setVisible(False)
        if self.opersys==0:
            path=F4G.get_desktop_path()
        else:
             path=''
        return path
    
    
    def slider_deck(self,widget, page_index):
          self.page_offset+=1
          if self.slider_name[page_index][0] != 'None':
            if page_index in self.decimal_tenths:
                initial_slider=(self.init_slider[page_index][0]/10)
                self.current_label = QLabel(self.slider_name[page_index][0] + ': ' + str(initial_slider), widget)
            else:
                self.current_label = QLabel(self.slider_name[page_index][0] + ': ' + str(self.init_slider[page_index][0]), widget)
            self.current_slider = QSlider(Qt.Horizontal, widget)
            self.current_slider.valueChanged[int].connect(self.on_slider_change)
            self.current_slider.setMinimum(self.Min_slider[page_index][0])
            self.current_slider.setTickInterval(10)
            self.current_slider.setMaximum(self.Max_slider[page_index][0])
            self.current_slider.setValue(self.init_slider[page_index][0])
            self.current_slider.setEnabled(True)
            self.current_label.setEnabled(True)
            self.current_label.setGeometry(100, 60, 200, 60)
            self.current_slider.setGeometry(100, 100, 200, 60)

            if len(self.slider_name[page_index])>1:
                if page_index in self.decimal_hundreths_2:
                    initial_slider=(self.init_slider[page_index][1]/100)
                    self.current_label_2 = QLabel(self.slider_name[page_index][1] + ': ' + str(initial_slider), widget)
                else:
                    self.current_label_2 = QLabel(self.slider_name[page_index][1] + ': ' + str(self.init_slider[page_index][1]), widget)
                self.current_slider_2 = QSlider(Qt.Horizontal, widget)
                self.current_slider_2.valueChanged[int].connect(self.on_slider_change_2)
                self.current_slider_2.setMinimum(self.Min_slider[page_index][1])
                self.current_slider_2.setMaximum(self.Max_slider[page_index][1])
                self.current_slider_2.setTickInterval(10)
                self.current_slider_2.setValue(self.init_slider[page_index][1])
                self.current_label_2.setGeometry(320, 60, 200, 60)
                self.current_slider_2.setGeometry(320, 100, 200, 60)


            if len(self.slider_name[page_index])>2:
                self.current_label_3 = QLabel(self.slider_name[page_index][2] + ': ' + str(self.init_slider[page_index][2]), widget)
                self.current_slider_3 = QSlider(Qt.Horizontal, widget)
                self.current_slider_3.valueChanged[int].connect(self.on_slider_change_3)
                self.current_slider_3.setMinimum(self.Min_slider[page_index][2])
                self.current_slider_3.setMaximum(self.Max_slider[page_index][2])
                self.current_slider_3.setTickInterval(10)
                self.current_slider_3.setValue(self.init_slider[page_index][2])
                self.current_label_3.setGeometry(540, 60, 200, 60)
                self.current_slider_3.setGeometry(540, 100, 200, 60)
        
    
    @ pyqtSlot()
    def select_type_of_model(self): 
            self.thread.model()
            self.delete_widgets(self.select_model_button)
            if self.value==1:
                self.load_button.setVisible(True)

    @ pyqtSlot()
    def pause_signal(self):
         self.thread.stop_timer()

    @ pyqtSlot()
    def resume_signal(self):
         self.thread.start_timer()
 
    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
            self.display_vid1.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot(QImage)
    def displayFrame2(self, Image):
            self.display_vid2.setPixmap(QPixmap.fromImage(Image))

    @ pyqtSlot()
    def get_initial_data(self): 
            self.load_button.setVisible(False)
            self.comboBox2.setVisible(False)
            self.comboBox1.setVisible(False)
            self.Box_label.setVisible(False)
            self.Box_label2.setVisible(False)
            self.thread.initial_file_opening()
            self.next_btn.setVisible(True)

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
        param=None
        index=self.current_index
        if self.current_index in self.apply_perm_function:
             self.thread.apply_changes()
        if self.slider_name[self.current_index][0] != 'None':
            param=self.init_slider[self.current_index]
        if self.current_index in self.method_indx:
            param=np.append(param, self.init_method[self.current_index-2])
        self.thread.get_preset_values(index, param)

    @pyqtSlot(str)
    def sel_method(self,method):
        self.thread.change_method(method)

# *** New
    @pyqtSlot()
    def eeg_folder(self):
        path = self.path_selector()
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("All Files (*);;NetCDF files (*.nc)")
        if path:
            file_dialog.setDirectory(path) 

        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                file_path = selected_files[0]
                self.thread.eeg_folder_path(file_path)
                self.eeg_button.setVisible(False)
                self.velocity_button.setVisible(True)

    @pyqtSlot()
    def velocity_folder(self):
        path = self.path_selector()
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("All Files (*);;NetCDF files (*.nc)")
        if path:
            file_dialog.setDirectory(path)
        
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                    file_path = selected_files[0]
                    self.thread.velocity_folder_path(file_path)
                    self.velocity_button.setVisible(False)
                    self.next_btn.setVisible(True)

    @pyqtSlot()
    def on_slider_change(self):
            current_value = self.current_slider.value()
            self.thread.temp_mod_1(current_value)
            if self.current_index in self.decimal_tenths:
                current_value=(current_value/10)
                self.current_label.setText("{0}: {1}".format(self.slider_name[self.current_index][0], str(current_value)))
                 
    @pyqtSlot()
    def on_slider_change_2(self):
            current_value_2=self.current_slider_2.value()
            try:
                if hasattr(self.thread, 'temp_mod_2'):
                    self.thread.temp_mod_2(current_value_2)
                    if self.current_index in self.decimal_hundreths_2:
                            current_value_2=(current_value_2/100)
                    self.current_label_2.setText("{0}: {1}".format(self.slider_name[self.current_index][1], str(current_value_2)))
            except AttributeError as e:
                 print('') # this is just here for initialization
    @pyqtSlot()
    def on_slider_change_3(self):
        current_value_3=self.current_slider_3.value()
        try:
                if hasattr(self.thread, 'temp_mod_3'):
                    self.thread.temp_mod_3(current_value_3)
                    self.current_label_3.setText("{0}: {1}".format(self.slider_name[self.current_index][2], str(current_value_3))) 
        except AttributeError as e:
                 print('') # this is just here for initialization


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())