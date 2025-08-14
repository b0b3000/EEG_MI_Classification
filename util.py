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
import seaborn as sns
from scipy.signal import stft
from pyriemann.utils.viz import plot_confusion_matrix
import pandas as pd
import pywt
from pyriemann.estimation import Covariances

DATASET_LOCATION = "/Users/bobbeashel/Desktop/CITS4010/Project/data/"
#DATASET_LOCATION = "/Users/bobbeashel/Desktop/CITS4010/Project/data/001-2014"

def convert_wavelet(X_train, X_test, sample_rate, num_frequencies):
    samples_per_trial = X_train.shape[2]
    fmin = 2
    fmax = sample_rate/2
    wavelet = "cmor3-3"

    freqs = np.linspace(fmin, fmax, num_frequencies)
    scales = pywt.scale2frequency(wavelet, 1.0) * sample_rate / freqs #scales are analogous to frequency, but not exactly the same. Used for wavelet

    def conversion_helper(X):
        X_shape = X.shape
        converted = np.zeros((X.shape[0], X.shape[1], num_frequencies, X.shape[2])) # num trials, num_channels, num frequencies, num samples
        for i in range(X_shape[0]):
            for j in range(X_shape[1]):
                    coef, _ = pywt.cwt(X[i][j], scales, wavelet, sampling_period=1/sample_rate)
                    power = np.abs(coef) ** 2
                    converted[i][j] = power
        return converted

    X_train_converted = conversion_helper(X_train)
    X_test_converted = conversion_helper(X_test)

    # TEMP vvvv
    times = np.arange(samples_per_trial) / sample_rate #500/250 = 2
    plt.figure(figsize=(10, 6))
    plt.contourf(times, freqs, X_train_converted[0][0], levels=100, cmap='viridis')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time-Frequency Representation (Wavelet Transform)')
    plt.colorbar(label='Power')
    plt.tight_layout()
    plt.show()

    return X_train_converted, X_test_converted


def plot_all_predicted_probabilities(probs, class_names=None):
    """
    Plot the distribution of maximum predicted probabilities for each instance,
    including both individual dots (stripplot) and summary (boxplot).
    
    Args:
        probs: np.ndarray of shape (n_samples, n_classes)
    """
    max_probs = probs.max(axis=1)  # get top-1 probability for each sample

    plt.figure(figsize=(12, 5))
    
    # Create both stripplot and boxplot on the same axis
    sns.stripplot(x=max_probs, orient='h', jitter=0.2, alpha=0.5, color='dodgerblue', label='Individual Samples')
    sns.boxplot(x=max_probs, orient='h', color='lightgray', width=0.3, fliersize=0, linewidth=1)

    plt.xlabel('Top-1 Predicted Probability')
    plt.title('Top-1 Confidence Distribution (with Boxplot)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend()
    plt.show()

def xdawnrg(X_train, X_test, Y_train, Y_test, chans, samples, names):
    ############################# xDAWN + RG Portion ##############################

    # code is taken from PyRiemann's ERP sample script, which is decoding in 
    # the tangent space with a logistic regression

    n_components = 2  # pick some components

    # set up sklearn pipeline
    clf = make_pipeline(XdawnCovariances(n_components),
                        TangentSpace(metric='riemann'),
                        LogisticRegression())
    
    preds_rg     = np.zeros(len(Y_test))

    # reshape back to (trials, channels, samples)
    X_train_reshaped      = X_train.reshape(X_train.shape[0], chans, samples)
    X_test_reshaped       = X_test.reshape(X_test.shape[0], chans, samples)

    # train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
    # labels need to be back in single-column format
    clf.fit(X_train_reshaped, Y_train.argmax(axis = -1))
    preds_rg     = clf.predict(X_test_reshaped)

    # Printing the results
    acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
    print("Classification accuracy xDAWN + RG: %f " % (acc2))

    plt.figure(1)
    plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'xDAWN + RG')

    plt.show()

def convert_stft(X_train, X_test, sample_rate, segment_len=64, sample_overlap=32, boundary="zeros", padding=True):
    def compute_stft(X,dataset):
        trial_count, channel_count, samples_per_trial = X.shape

        freq_bins_centers, time_window_centers, sample_stft = stft(X[0, 0], fs=sample_rate, nperseg=segment_len, noverlap=sample_overlap, boundary=boundary, padded=padding)
        #freq_bins_centers: 1D array of all frequency bin center frequencies
        #time_window_centers: 1D array of all time window center times for STFT windows (in seconds)
        #a: Test 2D array of STFT for Channel 0 Sample 0
        # We do this so we can know the values of num windows and num freq bins
        
        visualise_sample_stft(freq_bins_centers, time_window_centers, sample_stft,dataset)

        #Initialise 4D array to store a 2D STFT array for each trial and channel combination
        X_time_freq = np.empty((trial_count, channel_count, len(freq_bins_centers), len(time_window_centers)), dtype=np.float32)

        for trial in range(trial_count):
            for channel in range(channel_count):
                _, _, Z_temp = stft(X[trial, channel], fs=sample_rate, nperseg=segment_len, noverlap=sample_overlap, boundary=boundary, padded=padding)
                X_time_freq[trial, channel] = np.abs(Z_temp)
                #freq_bins_centers, time_window_centers are the same each time, only need to collect once

        return X_time_freq, freq_bins_centers, time_window_centers

    X_train_tf, freq_bins_centers, time_window_centers = compute_stft(X_train, "Training Set")
    X_test_tf, _, _ = compute_stft(X_test, "Testing Set")
    #freq_bins_centers, time_window_centers are the same each time, only need to collect once

    return X_train_tf, X_test_tf, freq_bins_centers, time_window_centers

def visualise_sample_stft(freqs, times, sample_stft, dataset="Training Set"):
    sample_stft = np.abs(sample_stft)  
    print("Each sample + channel has", len(freqs), "frequency bins with", len(times), "windows each.")

    fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    # --- Left subplot: STFT spectrogram ---
    pcm = axs[0].pcolormesh(times, freqs, sample_stft, shading='gouraud')
    axs[0].set_ylabel('Frequency (Hz)')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_title('STFT Magnitude — Trial 0, Channel 0 ' + dataset)

    # Vertical time grid
    for t in times:
        axs[0].axvline(x=t, color='gray', linestyle='--', linewidth=0.3)

    # Horizontal frequency grid
    for f in freqs:
        axs[0].axhline(y=f, color='gray', linestyle='--', linewidth=0.3)

    # Add colorbar
    fig.colorbar(pcm, ax=axs[0], label='Magnitude')

    # --- Right subplot: Frequency profile at a single time window ---
    axs[1].bar(freqs, sample_stft[:, 1], align='center', color='skyblue')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_title(f'STFT at {times[1]:.2f}s (Trial 0 Channel 0) '+ dataset)

    plt.tight_layout()
    plt.show()

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

def plot_prediction_confidence(probs, k=1.2):

    """
    Plots a strip plot (dot plot) of Top1/Top2 confidence ratios for all samples, capped at 3.
    
    Parameters:
    - probs: 2D numpy array of shape (num_samples, num_classes)
    """
    sorted_probs = -np.sort(-probs, axis=1)
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1]

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        confidence_ratio = np.where(top2 != 0, top1 / top2, np.inf)

    # Cap extreme values at 3 for visualization clarity
    confidence_ratio = np.clip(confidence_ratio, a_min=None, a_max=5)

    # Seaborn style
    sns.set(style="whitegrid")

    # Plot
    plt.figure(figsize=(10, 5))
    # Highlight region from 1 to k
    #plt.axvspan(1, k, color='red', alpha=0.2, label=f'Uncertain Region (1–{k})')
    sns.histplot(confidence_ratio, kde=True, bins=30, color='skyblue')
    plt.title("Histogram + KDE of Confidence Ratios (Top1 / Top2)")
    plt.xlabel("Confidence Ratio")
    plt.ylabel("Frequency")

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
def exponential_moving_standardize(train_segments, test_segments, decay=0.999, init_block_size=1000):
    """
    Performs electrode-wise exponential moving standardization on EEG data.
    
    Parameters
    ----------
    train_segments : np.ndarray
        Shape: (n_trials, n_channels, n_samples)
    test_segments : np.ndarray
        Shape: (n_trials, n_channels, n_samples)
    decay : float
        EMA decay factor (default: 0.999)
    init_block_size : int
        Number of initial samples to use for computing starting mean and variance
        from the training set (per channel).
    
    Returns
    -------
    standardized_train, standardized_test : np.ndarray
        Standardized versions of train_segments and test_segments.
    """
    
    # Concatenate trials into continuous data for EMA computation
    train_cont = np.concatenate(train_segments, axis=1)  # shape: (n_channels, total_samples)
    test_cont  = np.concatenate(test_segments, axis=1)

    n_channels, n_total_train_samples = train_cont.shape
    _, n_total_test_samples = test_cont.shape

    # Initialize mean and variance from first `init_block_size` samples of TRAIN ONLY
    init_mean = np.mean(train_cont[:, :init_block_size], axis=1, keepdims=True)
    init_var  = np.var(train_cont[:, :init_block_size], axis=1, keepdims=True)

    # Allocate outputs
    train_out = np.zeros_like(train_cont)
    test_out  = np.zeros_like(test_cont)

    # Initialize running stats
    mean_t = init_mean.copy()
    var_t  = init_var.copy()

    # ---- Standardize TRAIN ----
    for t in range(n_total_train_samples):
        x_t = train_cont[:, t:t+1]  # shape: (n_channels, 1)
        mean_t = (1 - decay) * x_t + decay * mean_t
        var_t  = (1 - decay) * (x_t - mean_t) ** 2 + decay * var_t
        train_out[:, t:t+1] = (x_t - mean_t) / np.sqrt(var_t + 1e-8)

    # ---- Standardize TEST ----
    # Carry over mean_t and var_t from the end of TRAIN
    for t in range(n_total_test_samples):
        x_t = test_cont[:, t:t+1]
        mean_t = (1 - decay) * x_t + decay * mean_t
        var_t  = (1 - decay) * (x_t - mean_t) ** 2 + decay * var_t
        test_out[:, t:t+1] = (x_t - mean_t) / np.sqrt(var_t + 1e-8)

    # Reshape back to original trial structure
    def split_trials(standardized, original_segments):
        result = []
        idx = 0
        for trial in original_segments:
            n_samp = trial.shape[1]
            result.append(standardized[:, idx:idx+n_samp])
            idx += n_samp
        return np.array(result)

    standardized_train = split_trials(train_out, train_segments)
    standardized_test  = split_trials(test_out, test_segments)

    return standardized_train, standardized_test

def bci_2a_helper(file_names, tmin, tmax, chans,bandpass, mode, amp_mag, baseline):
    if file_names:
        all_segments = []
        all_labels = []
        if mode == "gdf":
            directory = DATASET_LOCATION + "BCICIV_2a_gdf/" #.gdf files locations
            for i, file_name in enumerate(file_names):
                data_path = directory + file_name + ".gdf"
                labels_path = directory + "true_labels/" + file_name + ".mat"
                raw = mne.io.read_raw_gdf(data_path, preload=True, verbose=0)

                ################################################ BANDPASS ####################################
                
                raw.filter(bandpass[0],bandpass[1], fir_design='firwin', skip_by_annotation='edge', verbose=0)
                #raw.filter(2, None, method='iir') 

                ###############################################################################################

                events, _ = mne.events_from_annotations(raw, event_id = {
                    '769': 1,   # left hand
                    '770': 2,   # right hand
                    '771': 3,   # foot
                    '772': 4,   # tongue
                    '783': 5    #unknown (eval sets)
                }, verbose = 0)

                #Epoch data into windowed trials
                epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=baseline, preload=True,verbose=0)

                #Get the signal data from the EEG channels of epoch
                all_segments.append(epochs.get_data()[:,:chans,:-1]) # i did this -1 because the samples was always exactly 1 too high.

                labels_raw = loadmat(labels_path)
                labels_raw = labels_raw["classlabel"].reshape(-1) - 1 #Change from 1,2,3,4 to 0,1,2,3 because the EEGNet model likes it
                all_labels.append(labels_raw)
    
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

def get_bci_2a(file_names_training, file_names_testing, bandpass, tmin, tmax, mode, amp_mag, baseline):
    sample_rate = 250 #From BCI Dataset description
    kernels, chans = 1, 22 # There are actually 25 channels, but we only want to retain 22, as 3 are EOG
    names        = ['left', 'right', 'foot', 'tongue']

    samples = int((tmax - (tmin)) * sample_rate)
    train_segments, train_labels = bci_2a_helper(file_names_training, tmin, tmax, chans, bandpass, mode, amp_mag, baseline)
    test_segments, test_labels = bci_2a_helper(file_names_testing, tmin, tmax, chans, bandpass, mode, amp_mag, baseline)

    train_segments, test_segments = exponential_moving_standardize(train_segments, test_segments)
    
    return(train_segments, train_labels, test_segments, test_labels, chans, kernels, samples, names, sample_rate)

def bci_2b_helper(file_names, tmin, tmax, chans,bandpass, mode, amp_mag, baseline):
    if file_names:
        all_segments = []
        all_labels = []
        if mode == "gdf":
            directory = DATASET_LOCATION + "BCICIV_2b_gdf/" #.gdf files locations
            for i, file_name in enumerate(file_names):
                data_path = directory + file_name + ".gdf"
                labels_path = directory + "true_labels/" + file_name + ".mat"
                raw = mne.io.read_raw_gdf(data_path, preload=True, verbose=0)

                ################################################ BANDPASS ####################################
                
                raw.filter(bandpass[0],bandpass[1], fir_design='firwin', skip_by_annotation='edge', verbose=0)
                #raw.filter(2, None, method='iir') 

                ###############################################################################################

                events, _ = mne.events_from_annotations(raw, event_id = {
                    '769': 1,   # left hand
                    '770': 2,   # right hand
                    '783': 3   #unknown (eval sets)
                }, verbose = 0)

                #Epoch data into windowed trials
                epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=baseline, preload=True,verbose=0)

                #Get the signal data from the EEG channels of epoch
                all_segments.append(epochs.get_data()[:,:chans,:-1]) # i did this -1 because the samples was always exactly 1 too high.

                labels_raw = loadmat(labels_path)
                labels_raw = labels_raw["classlabel"].reshape(-1) - 1 #Change from 1,2,3,4 to 0,1,2,3 because the EEGNet model likes it
                all_labels.append(labels_raw)
    
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

def get_bci_2b(file_names_training, file_names_testing, bandpass, tmin, tmax, mode, amp_mag, baseline):
    sample_rate = 250 #From BCI Dataset description
    kernels, chans = 1, 3 # There are actually 6 channels, but we only want to retain 3, as 3 are EOG
    names        = ['left', 'right']

    samples = int((tmax - (tmin)) * sample_rate)
    train_segments, train_labels = bci_2b_helper(file_names_training, tmin, tmax, chans, bandpass, mode, amp_mag, baseline)
    test_segments, test_labels = bci_2b_helper(file_names_testing, tmin, tmax, chans, bandpass, mode, amp_mag, baseline)

    return(train_segments, train_labels, test_segments, test_labels, chans, kernels, samples, names, sample_rate)

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