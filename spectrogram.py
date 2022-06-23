from __future__ import annotations
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
import pickle
import os

import librosa.display
import librosa

DATA_PATH = "./mit_data/"

transformed_signal = []
dt = 0.001

# 경로에 폴더가 없으면 폴더 만들기
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

record_list = []
pickle_input = dict()
X, y = [], []

print("[INFO] Read records file from ", DATA_PATH)
with open(DATA_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

for i in tqdm(range(len(record_list))):
    temp_path = DATA_PATH + "mit" + record_list[i] + ".pkl"
    with open(temp_path, 'rb') as f:
        pickle_input = pickle.load(f)
        for i in range(len(pickle_input[0])):
            X.append(pickle_input[0][i])

        for i in range(len(pickle_input[1])):
            check_ann = pickle_input[1][i]
            temp_ann_list = list()
            if check_ann == "N":            # Normal
                temp_ann_list.append(0)

            elif check_ann == "S":          # Supra-ventricular
                temp_ann_list.append(1)

            elif check_ann == "V":          # Ventricular
                temp_ann_list.append(2)

            elif check_ann == "F":          # False alarm
                temp_ann_list.append(3)

            else:                           # Unclassed 
                temp_ann_list.append(4)
            
            y.append(temp_ann_list)

PATH = "./spectrogram_plt/"
X = np.array(X)
y = np.array(y)

for cnt in tqdm(range(len(X))):
    fig = plt.figure(figsize=[10, 10])
    plt.interactive(False)
    
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)  
    
    stft = np.abs(librosa.stft(X[cnt], n_fft=428, win_length=64))
    stft = librosa.amplitude_to_db(stft, ref=np.max)
    stft = librosa.display.specshow(stft, y_axis='log', x_axis='time', ax=ax)
    
    annotation = y[cnt][0]

    if annotation == 0:
        createDirectory(PATH + "N/")
        SAVE_PATH = PATH + "N/" + str(cnt) + "N_" + ".png"
    
    elif annotation == 1:
        createDirectory(PATH + "S/")
        SAVE_PATH = PATH + "S/" + str(cnt) + "S_" + ".png"
    
    elif annotation == 2:
        createDirectory(PATH + "V/")
        SAVE_PATH = PATH + "V/" + str(cnt) + "V_" + ".png"
    
    elif annotation == 3:
        createDirectory(PATH + "F/")
        SAVE_PATH = PATH + "F/" + str(cnt) + "F_" + ".png"

    else:
        createDirectory(PATH + "Q/")
        SAVE_PATH = PATH + "Q/" + str(cnt) + "Q_" + ".png"

    plt.savefig(SAVE_PATH)
    plt.clf()
    plt.close()

plt.show()
