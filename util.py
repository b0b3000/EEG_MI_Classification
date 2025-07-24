from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import mne
from mne import io
from mne.datasets import sample
from models import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from collections import Counter

DATASET_LOCATION = "/Users/bobbeashel/Desktop/CITS4010/Project/data/"
#DATASET_LOCATION = "/Users/bobbeashel/Desktop/CITS4010/Project/data/001-2014"

def plot_predicted_probs(probs, num_samples_to_plot):
    subset = probs[:num_samples_to_plot]

    labels = [f'Sample {i}' for i in range(num_samples_to_plot)]
    classes = [f'Class {i}' for i in range(probs.shape[1])]

    bottom = np.zeros(num_samples_to_plot)

    plt.figure(figsize=(12, 6))
    for i in range(probs.shape[1]):
        plt.bar(labels, subset[:, i], bottom=bottom, label=classes[i])
        bottom += subset[:, i]

    plt.ylabel('Probability')
    plt.title(f'Class Probabilities (First {num_samples_to_plot} Samples)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def get_mne_dataset():
    kernels, chans, samples = 1, 60, 151
    names = ['audio left', 'audio right', 'vis left', 'vis right']
    tmin, tmax = -0., 1

    # while the default tensorflow ordering is 'channels_last' we set it here
    # to be explicit in case if the user has changed the default ordering
    K.set_image_data_format('channels_last')

    ##################### Read the data ######################

    data_path = sample.data_path()

    # Set parameters and read data
    raw_fname = str(data_path) + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = str(data_path) + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
    
    event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

    # Setup for reading the raw data
    raw = io.Raw(raw_fname, preload=True, verbose=False)
    events = mne.read_events(event_fname)

    ##################### Preprocess the data ######################

    raw.filter(2, None, method='iir')  # replace baselining with high-pass

    raw.info['bads'] = ['MEG 2443']  # set bad channels

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    ##################### Epoch, split and reshape the data ######################

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks, baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1] 
    labels = np_utils.to_categorical(labels-1) # -1 Because values start at 1 rather than 0

    # scale due to scaling sensitivity in deep learning
    X = epochs.get_data() * 1000 

    return(X, labels, chans, kernels, samples, names)

def bci_2a_helper(file_names, tmin, tmax, chans,bandpass, mode, amp_mag):
    if file_names:
        all_segments = []
        all_labels = []
        if mode == "gdf":
            directory = DATASET_LOCATION + "BCICIV_2a_gdf/" #.gdf files locations
            for i, file_name in enumerate(file_names):
                data_path = directory + file_name + ".gdf"
                labels_path = directory + "true_labels/" + file_name + ".mat"
                raw = mne.io.read_raw_gdf(data_path, preload=True)

                ################################################ BANDPASS ####################################
                
                raw.filter(bandpass[0],bandpass[1], fir_design='firwin', skip_by_annotation='edge')
                #raw.filter(2, None, method='iir') 

                ###############################################################################################

                events, _ = mne.events_from_annotations(raw, event_id = {
                    '769': 1,   # left hand
                    '770': 2,   # right hand
                    '771': 3,   # foot
                    '772': 4,   # tongue
                    '783': 5    #unknown (eval sets)
                })

                #Epoch data into windowed trials
                epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True)

                #Get the signal data from the EEG channels of epoch
                all_segments.append(epochs.get_data()[:,:chans,:-1]) # i did this -1 because the samples was always exactly 1 too high.

                labels_raw = loadmat(labels_path)
                labels_raw = labels_raw["classlabel"].reshape(-1) - 1 #Change from 1,2,3,4 to 0,1,2,3 because the EEGNet model likes it
                all_labels.append(labels_raw)
    

        elif mode == "mat":
            directory = DATASET_LOCATION + "001-2014/" #.mat files locations
            
            for i, file_name in enumerate(file_names):

                ####################### GET FILE #######################
                print("\n", file_name)
                path = directory + file_name + ".mat"
                data_dict = loadmat(path)
                data = data_dict['data'].squeeze()[3:]# First 3 are EOG calibration, not MI
                
                k, cum_time = 0,0
                trials_list = []
                for t in data:             
                    k += 1
                    cum_time += t['X'][0,0].shape[0]
                    trials = t['trial'][0, 0][:,0]
                    adjusted_trials = trials + cum_time
                    trials_list.append(adjusted_trials)
                trials = np.concatenate(trials_list)

                X_list = [t['X'][0, 0][:, :22] for t in data]
                X = np.concatenate(X_list, axis=0)

                sample_rate = data[0]['fs'][0][0][0][0]

                y_list_i = [t['y'][0, 0][:,0] for t in data]
                y_i = np.concatenate(y_list_i)
                y_i = y_i - 1

                raw = mne.io.RawArray(X.T, mne.create_info(ch_names=[f"EEG{j+1}" for j in range(X.shape[1])], sfreq=sample_rate, ch_types="eeg"))
                raw.filter(4, 40, method='iir') 

                events = np.column_stack([trials,np.zeros_like(trials), y_i])
                epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, reject_by_annotation=False)
                print(Counter(tuple(r) for r in epochs.drop_log))

                all_segments.append(epochs.get_data()[:,:chans,:-1])
                all_labels.append(epochs.events[:, 2].astype(int))
                print(all_segments[i].shape)
    

        #combining all elements of tracked list
        labels = np.concatenate(all_labels, axis=0)
        segments= np.concatenate(all_segments, axis=0)
        
        ############ AMPLITUDE MAG ##################################
        
        segments = segments * amp_mag

        ##########################################################

        # NP formatting
        segments = np.array(segments)
        labels = np.array(labels)

        return segments, labels
    else:
        return None, None

def get_bci_2a(file_names_training, file_names_testing, bandpass, tmin, tmax, mode, amp_mag):
    sample_rate = 250 #From BCI Dataset description
    kernels, chans = 1, 22 # There are actually 25 channels, but we only want to retain 22, as 3 are EOG
    names        = ['left', 'right', 'foot', 'tongue']

    samples = int((tmax - (tmin)) * sample_rate)
    train_segments, train_labels = bci_2a_helper(file_names_training, tmin, tmax, chans, bandpass, mode, amp_mag)
    test_segments, test_labels = bci_2a_helper(file_names_testing, tmin, tmax, chans, bandpass, mode, amp_mag)

    return(train_segments, train_labels, test_segments, test_labels, chans, kernels, samples, names)

#Function plots 1 epoch
def plot_epoch_with_event(epoch, sfreq, tmin=0.0, channel_names=None, title=None):

    n_channels, n_times = epoch.shape
    times = tmin + np.arange(n_times) / sfreq

    offset = np.max(np.abs(epoch)) * 1.2
    fig, ax = plt.subplots(figsize=(8, n_channels * 0.3))
    for ch in range(n_channels):
        ax.plot(times, epoch[ch] + ch * offset, label=(channel_names[ch] if channel_names else None))
    # mark event at t=0
    ax.axvline(0.0, color='k', linestyle='--', linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_yticks(np.arange(n_channels) * offset)
    if channel_names:
        ax.set_yticklabels(channel_names)
    else:
        ax.set_yticklabels([f'Ch {i}' for i in range(n_channels)])
    ax.set_title('Single Epoch with Event Onset (t=0)')
    ax.grid(True, axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.title(title)
    plt.show()