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
import scipy.signal as sig

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
        self.subject_dict={'/SL01':1,'/SL02':2,'/SL03':3,'/SL04':4, '/SL05':5,'/SL06':6,'/SL07':7,'/SL08':8}
        self.subject=self.get_subject(1)
        self.data_size_dictionary= {'Single subject: One session':0,'Single subject: Three sessions':1 , 'Entire dataset':2}
        self.training_dataset=None
        self.test_dataset=None
        self.data_selected=self.data_size_dictionary['Single subject: One session']
        self.eeg_raw=None
        self.slider_value=None
        self.slider_value_3=None
        self.slider_value_2=None
        self.current_index=0
        self.fs=100 # frequency of eeg
        self.decimal_tenths=[0] # indices where decimals must be used
        self.decimal_tenths_2=[1]
        self.parameters=None
        self.subject_selected=0
        self.function_methods=[3] # index where methods are passed
        self.method=None
        self.Update_previous=[2] # index where previous eeg is updated

### Functions below are for data loading
    def set_folder_path(self, over_all_path):
        Folder_path = os.path.abspath(os.path.join(over_all_path))
        # F4G.set_file_path(Folder_path)
        print(Folder_path)
        
    def initial_file_opening(self, **kwargs):
        self.data_selection()
        self.data_for_viz()

# Below functions are for data selection
    def data_for_viz(self): 
        if self.data_selected == 2:
            self.eeg_data = self.eeg_array[self.subject_selected]
        elif self.data_selected in [0, 1]:  
            self.eeg_data = self.eeg_array
        self.eeg_previous = self.eeg_data.isel(Trial=0).copy()
        print(self.eeg_previous.Channel.shape)
    
    def data_selection(self): # might be better to use a more randomized method but this is okay for now
        if self.data_selected==0:
            self.trails=[1]
        else: 
            self.trails=[1, 2, 3]
        if self.data_selected in [0,1]:
            self.eeg_array=F4G.Eeg_data_function(self.subject, self.trails, [2,17])
            self.joint_array=F4G.joint_data_function(self.subject, self.trails, [2,17])     
        elif self.data_selected==2:
            self.get_all_data()

    def get_all_data(self):
            self.eeg_array=np.empty(len(self.subject_dict.keys()))
            self.joint_array=np.empty(len(self.subject_dict.keys()))
            for  indx, subject in enumerate(self.subject_dict.keys()):
                self.eeg_array[indx]=F4G.Eeg_data_function(subject, [1], [2,17])
                self.joint_array[indx]=F4G.joint_data_function(subject,[1], [2,17])

    def change_subject(self, subject):
        self.subject_selected=subject # Numerical
        self.subject=self.get_subject(subject) # Dictionary value
        print(self.subject)

    def get_subject(self,value):
        for key, val in self.subject_dict.items():
            if val == value:
                return key
        return None  
                  
    def select_data_size(self, value):
         self.data_selected=self.data_size_dictionary[value]
### Functions below handle perminant changes to the EEG data

    def apply_changes_2(self):  # 
        self.generate_parameter_array()
        save_instance = Save_changes(self.EEG_array, self.current_index, self.data_selected, **self.parameters)
        save_instance.start()
        save_instance.wait()
        if save_instance.EEG_array is not None:
            self.EEG_array = save_instance.EEG_array
        else:
            print('Failed to obtain data from: Save_changes')
        self.parameters=None
        self.slider_value=0
        self.slider_value_2=None
        self.slider_value_3=None
        self.slider_value_4=None
        self.methods=None
        return self.EEG_array

    def generate_parameter_array(self):
        self.parameters = []  # Initialize the parameters list as an empty list
        if self.method is not None and not (self.current_index in self.function_methods):
            self.parameters.append(str(self.method))  # Convert method to string and append
        if self.slider_value != 0:
            self.parameters.append(str(self.slider_value))  # Convert slider_value to string and append
        if self.slider_value_2 is not None:
            self.parameters.append(str(self.slider_value_2))  # Convert slider_value_2 to string and append
        if self.slider_value_3 is not None:
            self.parameters.append(str(self.slider_value_3))  # Convert slider_value_3 to string and append
        if self.slider_value_4 is not None:
            self.parameters.append(str(self.slider_value_4))  # Convert slider_value_4 to string and append
        self.parameters = ','.join(self.parameters)  # Join the list elements with a comma
        if self.parameters:
            self.parameters = self.parameters.split(',')  # Split the joined string by commas
            for i, parameter in enumerate(self.parameters):
                try:
                    self.parameters[i] = float(parameter)  # Convert the number to a float
                except ValueError:
                    pass  # Ignore if conversion to float fails (e.g., for non-numeric values)
            if self.current_index in self.function_methods and self.method is not None:
                new_interval = (float(self.parameters[-1]), self.method)
                self.parameters[-1] = new_interval  # Convert interval_sliders to string and append
        print(self.parameters)

### Functions below are responsible for interactive graphing changes
    def function_controler(self):
            if self.current_index==1:
                input=self.eeg_previous.isel(Channel=12)
                print(input.shape)
                self.bandpass_filter(data=input,highcut=self.slider_value_3,lowcut=self.slider_value_2,fs=self.fs,order=self.slider_value)
                self.graph_band_pass()

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = sig.butter(order, [lowcut, highcut], btype='band', fs=fs)
        self.eeg_filtered = sig.filtfilt(b, a, data)
        self.graph_band_pass()
    
    def graph_band_pass(self):
        matplotlib.use('Agg')
        fig, ax = plt.subplots()
        plt.plot(self.eeg_previous.Recording.Time, self.eeg_previous.isel(Channel=12), label='Original Signal', alpha=0.7)
        plt.plot(self.eeg_previous.Time, self.eeg_filtered, label='Filtered Signal bandpass {}-{}Hz'.format(self.slider_value_2, self.slider_value_3), alpha=0.7)
        plt.title('EEG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
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
        self.handle_Display1(q_img)

### Functions below handles Display_video communication
    def play_eeg(self):
        if self.play_eeg_instance is not None and self.play_eeg_instance.isFinished:
            if self.eeg_previous is not None and len(self.eeg_previous) > 0:
                self.play_eeg_instance= Dispaly_video1(self.eeg_previous,self.spatial)
                self.play_eeg_instance.Display1.connect(self.handle_Display1)
                self.play_eeg_instance.start()
        elif self.play_eeg_instance is not None and self.play_eeg_instance.isRunning():
            self.play_eeg_instance.requestInterruption()
            if self.eeg_previous is not None and len(self.eeg_previous) > 0:
                self.play_eeg_instance = Dispaly_video1(self.eeg_previous,self.spatial)
                self.play_eeg_instance.Display1.connect(self.handle_Display1)
                self.play_eeg_instance.start()
        else:
            if self.eeg_previous is not None and len(self.eeg_previous) > 0:
                self.play_eeg_instance= Dispaly_video1(self.eeg_previous,self.spatial)
                self.play_eeg_instance.Display1.connect(self.handle_Display1)
                self.play_eeg_instance.start()

    def handle_Display1(self, qimage):
        self.Display1.emit(qimage)

### Functions below handle temporary MainWindow or Video interactions
    def get_preset_values(self, current_index, param):
        if current_index in self.Update_previous:
            self.data_for_viz() 
        self.current_index=current_index
        if param is not None:
            self.slider_value=param[0]
            if len(param)>1:
                self.slider_value_2=param[1]
                if len(param)>2:
                    self.slider_value_3=param[2]
            self.function_controler()

    def update_indx(self, index): 
         self.current_index=index
    
    def temp_mod(self, value): # takes value from on__change and adjusts value for functions
        self.slider_value = value
        if self.current_index in self.decimal_tenths:
            self.slider_value=value/10
        self.function_controler()
        
    def temp_mod_2(self, value_2):
        self.slider_value_2=value_2
        if self.current_index in self.decimal_tenths_2:
            self.slider_value_2=self.slider_value_2/10
        self.function_controler()

    def temp_mod_3(self, value_3):
        self.slider_value_3=value_3
        self.function_controler()

    def temp_method(self,method):
         self.method=method
         self.function_controler()

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

# Applies Transform
class Save_changes(QThread):  
        updateEEG_array=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, function_index=0, data_size=0, **param):
            super(Save_changes, self).__init__()
            self.eeg_data=eeg_data
            self.function_index=function_index
            self.data_size=data_size
            self.parameters=param
            self.fs=100
            self.value=None
            self.value_2=None
            self.value_3=None
            self.value_4=None
            self.trail=0

### Functions below are for class initiation

        def run(self):
            if self.parameters is not None and self.parameters !=[]:
                self.parameters_unpacking()
            if self.eeg_data is not None and len(self.eeg_data) > 0:
               self.apply_changes()
            else:
                print('Failed to pass data_array to Save_changes')

        def apply_changes(self):
            self.parameters_unpacking()
            if self.data_size==0:
                 self.single_by_channel()
            elif self.data_size==1:
                 self.three_sessions()
            elif self.data_size==2:
                 self.entire_data_set()
            print('Changes saved for ' + str(self.function_index))
            return self.eeg_data
                 
        def parameters_unpacking(self):
            if self.parameters is not None and len(self.parameters)!=0:
                print(self.parameters)
                self.value=self.parameters[0]
                print('1st passed value is '+ str(self.value))
            else:
                print('Parameters not obtained by Save_changes')
            if len(self.parameters)==2:
                self.value_2=self.parameters[1] 
            if len(self.parameters)==3:
                self.value_3=self.parameters[2]
            if len(self.parameters)==4:
                self.value_4=self.parameters[3]

### Following functions deal with applying functions to entire dataset one channel at a time
        def single_by_channel(self):
            for i in range(self.eeg_data.Channel.shape[0]):
                self.eeg_data.isel(Channel=i,Trial=self.trial)[:] = self.current_function(self.eeg_data.isel(Channel=i,Trial=self.trial)[:])

        def three_sessions(self):
             for j in range(3):
                  self.trial=j
                  self.single_session()

        def entire_data_set(self):
             for indx, xarray in enumerate(self.eeg_data):
                  new_xarray=self.three_sessions(xarray)
                  self.eeg_data[indx]=new_xarray       

### Following functions deals with calling and selecting specific functions to be applied to dataset

        def current_function(self, data):
            if self.function_index == 2:
                self.band_pass_filter(data)

        def band_pass_filter(self,data):
            b, a = sig.butter(self.value, [self.value_2, self.value_3], btype='band', fs=self.fs)
            self.eeg_filtered = sig.filtfilt(b, a, data)
            return self.eeg_filtered

# Applies Machine learning algorythim
class Apply_Predictor(QThread):  
        Predictions=pyqtSignal(pd.DataFrame)
        def __init__(self, eeg_data, joint_data,**kwargs):
            super(Apply_Predictor, self).__init__()
            self.eeg=eeg_data
            self.joint=joint_data               


# sends eeg imaging
class Dispaly_video1(QThread):
    Display1 = pyqtSignal(QImage)
    def __init__(self,eeg_data,spatial_data, **kwargs):
        super(Dispaly_video1, self).__init__()
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
            # print(self.eeg_data.shape)
            self.channels=len(self.spatial_data[0])
            # channel type from literature
            chan_type=['eeg']*self.channels
            self.ch_type=chan_type   

# sends images to second display
class Display_video2(QThread):
    MoveFrame = pyqtSignal(QImage) 
    def __init__(self, **kwargs):
        super(Display_video2, self).__init__()




# MainWindow handles all displays/interactions
class MainWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.current_index=0
        self.last_index=4
        self.decimal_tenths=[0] # indices where decimals must be used
        self.decimal_tenths_2=[1]
        self.opersys=0 # default value
        self.can_go_back=[0,0,0,1,0] # 0 : cannot go back, 1: can go back
        self.show_vid1=[0,1,0,1,0]
        self.Min_slider =[[0],[2,1,20],[0],[0],[0]]
        self.Max_slider = [[0],[10,10,50],[0],[0],[0]]
        self.slider_name =[['None'], ['order','High Pass', 'Low Pass'], ['None'], ['None'], ['None']]
        self.oper_type=['Windows:environment WSL2', 'Non-Windows: WSL2', 'Opsys not supported']
        self.select_subject={'Subject 1':1,'Subject 2':2, 'Subject 3':3,'Subject 4':4,
                                   'Subject 5':5 ,'Subject 6': 6,'Subject 7': 7,'Subject 8':8}
        self.data_size=['Single subject: One session','Single subject: Three sessions', 'Entire dataset']
        self.setWindowTitle("Insert title")
        self.init_slider=[[0],[5,5,40],[0], [0],[0]]
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
        self.next_btn.setVisible(False)

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
        # self.MidStack.setCurrentIndex(self.current_index)
        self.setWindowTitle("Welcome to eeg-based pridictor simulator")

    def init_new_widget(self):
            self.initial_data_loading()
            self.Bandpass_page()
            self.component_page()
            self.visual_denoising_verification()
            self.Button_manager=[[self.comboBox1,self.Box_label, self.comboBox2,self.Box_label2],[0],[0],[0],[0]]

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
        elif self.show_vid1[self.current_index]==0:
                 self.display_vid1.setVisible(False)
        if self.Button_manager[self.current_index][0]!=0:
             for button in self.Button_manager[self.current_index]:
                  button.setVisible(True)

    def preVisability_manager(self):
        if self.Button_manager[self.current_index][0]!=0:
             print(self.Button_manager[self.current_index])
             for button in self.Button_manager[self.current_index]:
                  button.setVisible(False)
         

    def initial_data_loading(self):
            # sets page 1 widget
            self.setWindowTitle("Selecting amount data")
            self.page_data_load=QWidget()

            # load data button (will be added to next button later)
            self.folder_button = QPushButton(self.page_data_load, text="Select folder with data")
            self.folder_button.setGeometry(600, 10, 200, 60)
            self.folder_button.clicked.connect(self.pass_rep_folder)
            self.folder_button.setVisible(False)


            # Selecting patient number
            self.comboBox1 = QComboBox(self)
            self.comboBox1.setGeometry(100, 600, 200, 60)
            self.comboBox1.addItems(self.select_subject.keys())  # Use the correct method drop-down options
            self.Box_label = QLabel('Subject on video', self)
            self.Box_label.move(100, 560)
            self.comboBox1.currentIndexChanged.connect(self.subject_sel)
            self.comboBox1.setVisible(False)

            # Selecting Dataset training size
            self.comboBox2 = QComboBox(self)
            self.comboBox2.setGeometry(300, 600, 200, 60)
            self.comboBox2.addItems(self.data_size)  # Use the correct method drop-down options
            self.Box_label2 = QLabel('Size of Dataset', self)
            self.Box_label2.move(300, 560)
            self.current_sel=self.data_size[self.comboBox2.currentIndex()]
            self.comboBox2.currentIndexChanged.connect(self.data_size_select)
            self.comboBox2.setVisible(False)

                        # Selecting os
            self.Opselect = QComboBox(self.page_data_load)
            self.Opselect.setGeometry(700, 100, 200, 60)
            self.Opselect.addItems(self.oper_type)  # Use the correct method drop-down options
            self.Ops_label = QLabel('Select os & env', self.page_data_load)
            self.Ops_label.move(700, 50)
            self.Opselect.currentIndexChanged.connect(self.on_os_changed)
            self.Box_label.setVisible(False)
            self.Box_label2.setVisible(False)
            # load data button (will be added to next button later)
            self.confirm_button = QPushButton(self.page_data_load, text="Press to confirm")
            self.confirm_button.setGeometry(700, 200, 200, 60)
            self.confirm_button.clicked.connect(self.path_selector)

            # load data button (will be added to next button later)
            self.load_button = QPushButton(self.page_data_load, text="Select to load data")
            self.load_button.move(400, 80)
            self.load_button.setGeometry(600, 80, 200, 60)
            self.load_button.clicked.connect(self.get_initial_data)
            self. page1=self.MidStack.addWidget(self.page_data_load)
            self.load_button.setVisible(False)
            print('intitializing widget 1st')
            return self. page1
    
    def Bandpass_page(self):
         self.bp_page=QWidget()
         self.slider_deck(self.bp_page,1)
         self.pagebp=self.MidStack.addWidget(self.bp_page)
         return self.pagebp
    
    def component_page(self):
        self.ca_page=QWidget()

         ## Add buttons
            # load data button (will be added to next button later)
        self.new_button = QPushButton(self.ca_page, text="Useless button")
        self.new_button.setGeometry(300, 10, 200, 60)

        self.pageca=self.MidStack.addWidget(self.ca_page)
        return self.pageca
        
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
    def on_os_changed(self):
        self.opersys=self.Opselect.currentIndex()

    def path_selector(self):
        self.folder_button.setVisible(True)
        self.Opselect.setVisible(False)
        self.Ops_label.setVisible(False)
        self.confirm_button.setVisible(False)
        print(self.opersys)
        if self.opersys==0:
            path=F4G.get_desktop_path()
        else:
             path=''
        print(path)
        return path
    
    def slider_deck(self,widget, page_index):
          # Creates first slider value
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
                if page_index in self.decimal_tenths_2:
                    initial_slider=(self.init_slider[page_index][1]/10)
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
                self.load_button.setVisible(True)
                self.load_button.setVisible(T)

    @ pyqtSlot(QImage)
    def displayFrame(self, Image):
            self.display_vid1.setPixmap(QPixmap.fromImage(Image))

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
        if self.slider_name[self.current_index][0] != 'None':
            param=self.init_slider[self.current_index]
        self.thread.get_preset_values(index, param)

    @pyqtSlot(int)
    def subject_sel(self, index):
        subject_name = list(self.select_subject.keys())[index]
        self.thread.change_subject(self.select_subject[subject_name])

    @pyqtSlot()
    def data_size_select(self):
        value = self.comboBox2.currentText()
        self.thread.select_data_size(value)

    @pyqtSlot()
    def pass_rep_folder(self):
        path=self.path_selector()
        folder_path = QFileDialog.getExistingDirectory(self, "Select RepositoryData folder", path)
        if not folder_path:  
            return    
        self.thread.set_folder_path(folder_path)  
        self.folder_button.setVisible(False)
        self.comboBox2.setVisible(True)
        self.comboBox1.setVisible(True)
        self.Box_label.setVisible(True)
        self.Box_label2.setVisible(True)
        self.load_button.setVisible(True)

    @pyqtSlot()
    def on_slider_change(self):
            current_value = self.current_slider.value()
            self.thread.temp_mod(current_value)
            if self.current_index in self.decimal_tenths:
                current_value=(current_value/10)
            self.current_label.setText("{0}: {1}".format(self.slider_name[self.current_index][0], str(current_value))) 

    @pyqtSlot()
    def on_slider_change_2(self):
            current_value_2=self.current_slider_2.value()
            self.thread.temp_mod_2(current_value_2)
            if self.current_index in self.decimal_tenths_2:
                    current_value_2=(current_value_2/10)
            self.current_label_2.setText("{0}: {1}".format(self.slider_name[self.current_index][1], str(current_value_2)))
    
    @pyqtSlot()
    def on_slider_change_3(self):
        current_value_3=self.current_slider_3.value()
        self.thread.temp_mod_3(current_value_3)
        self.current_label_3.setText("{0}: {1}".format(self.slider_name[self.current_index][2], str(current_value_3))) 
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())