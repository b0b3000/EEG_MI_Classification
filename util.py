from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import mne
from mne import io
from mne.datasets import sample
from models import EEGNet, EEGNet_Bob, ShallowConvNet, DeepConvNet, EEGNet_TF, EEGNet_Wavelet, EEGNet_Wavelet2, EEGNet_Wavelet3
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from collections import Counter
import seaborn as sns
from scipy.signal import stft
import pandas as pd
import pywt
from pyriemann.estimation import Covariances
from sklearn.utils import class_weight
import math

DATASET_LOCATION = "/Users/bobbeashel/Desktop/CITS4010/Project/data/"
#DATASET_LOCATION = "/Users/bobbeashel/Desktop/CITS4010/Project/data/001-2014"

def time_shift(x, max_shift=20):
    shift = np.random.randint(-max_shift, max_shift+1)
    return np.roll(x, shift, axis=1)  # shift along samples axis

def add_gaussian_noise(x, sigma=1e-6):
    noise = np.random.normal(0, sigma, size=x.shape).astype(x.dtype)
    return x + noise

def channel_dropout(x, k=1):
    x = x.copy()
    chans = x.shape[0]
    drop = np.random.choice(chans, size=k, replace=False)
    x[drop, :, 0] = 0.0  # zero out selected channels
    return x

def augment_trial(x, timeshift_prob, noise_prob, chan_dropout_prob):
    if np.random.rand() < timeshift_prob:
        x = time_shift(x, 20) # max shift = 20
    if np.random.rand() < noise_prob:
        x = add_gaussian_noise(x, 1e-6) #gaussian param
    if np.random.rand() < chan_dropout_prob:
        x = channel_dropout(x, 2) # dropout 2 channels randomly
    return x

def augment(X_train, Y_train, n_segments, timeshift_prob=0.5, noise_prob=0.5, chan_dropout_prob=0.3):
    n_samples = X_train.shape[2]
    
    segment_length = n_samples // n_segments

    augmented_X = []
    augmented_Y = []

    # keep original trials
    augmented_X.extend(X_train)
    augmented_Y.extend(Y_train)

    for i in range(len(X_train)):
        same_class_trials = np.where(np.argmax(Y_train, axis=1) == np.argmax(Y_train[i]))[0]

        new_trial = np.zeros_like(X_train[i])
        for seg_idx in range(n_segments):
            # choose a random trial from same class
            chosen_trial_idx = np.random.choice(same_class_trials)
            start = seg_idx * segment_length
            end = (seg_idx + 1) * segment_length
            new_trial[:, start:end, :] = augment_trial(X_train[chosen_trial_idx, :, start:end, :], timeshift_prob, noise_prob, chan_dropout_prob)
        augmented_X.append(new_trial)
        augmented_Y.append(Y_train[i])  # label stays the same

    augmented_X = np.array(augmented_X, dtype=np.float32)
    augmented_Y = np.array(augmented_Y, dtype=np.float32)

    return augmented_X, augmented_Y

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
'''
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
    plt.show()'''

def plot_predicted_probs(probs, num_samples_to_plot):
    subset = probs[:num_samples_to_plot]

    labels = [f'Sample {i}' for i in range(num_samples_to_plot)]
    classes = [f'Class {i}' for i in range(probs.shape[1])]

    colors = ["red", "blue", "green", "yellow"]  # rank-based colors

    plt.figure(figsize=(12, 6))

    for j in range(num_samples_to_plot):
        # Sort probabilities for this sample in descending order
        sorted_indices = np.argsort(subset[j])[::-1]
        sorted_probs = subset[j][sorted_indices]
        sorted_classes = [classes[k] for k in sorted_indices]

        bottom = 0
        for rank, (prob, cls) in enumerate(zip(sorted_probs, sorted_classes)):
            plt.bar(labels[j], prob, bottom=bottom,
                    color=colors[rank], label=cls if j == 0 else "")
            bottom += prob

    plt.ylabel('Probability')
    plt.title(f'Class Probabilities (First {num_samples_to_plot} Samples)')
    # Legend should show "Most likely", "2nd", etc.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", label="Most likely"),
        Patch(facecolor="blue", label="2nd most likely"),
        Patch(facecolor="green", label="3rd most likely"),
        Patch(facecolor="yellow", label="4th most likely")
    ]
    plt.legend(handles=legend_elements)
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

def apply_ica(raw):
    ica = mne.preprocessing.ICA(n_components=20)
    ica.fit(raw)
    raw.set_channel_types({ch: 'eog' for ch in raw.ch_names[-3:]})

    # Detect components correlated with EOG
    ica.exclude = ica.find_bads_eog(raw)[0]

    # Apply ICA
    raw = ica.apply(raw)

    # Now drop EOG before epochs
    raw.pick_types(eeg=True)

    return raw

def bci_2a_helper(file_names, tmin, tmax, chans,bandpass, mode, amp_mag, baseline, ica):
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

                ################################################## ICA ###################################

                if ica:
                    raw = apply_ica(raw)
                    
                ####################################################################################
                    
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

def get_bci_2a(file_names_training, file_names_testing, bandpass, tmin, tmax, mode, amp_mag, baseline, ica):
    sample_rate = 250 #From BCI Dataset description
    kernels, chans = 1, 22 # There are actually 25 channels, but we only want to retain 22, as 3 are EOG
    names        = ['left', 'right', 'foot', 'tongue']

    samples = int((tmax - (tmin)) * sample_rate)
    train_segments, train_labels = bci_2a_helper(file_names_training, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica)
    test_segments, test_labels = bci_2a_helper(file_names_testing, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica)

    train_segments, test_segments = exponential_moving_standardize(train_segments, test_segments)
    
    return(train_segments, train_labels, test_segments, test_labels, chans, kernels, samples, names, sample_rate)

def bci_2b_helper(file_names, tmin, tmax, chans,bandpass, mode, amp_mag, baseline, ica):
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

                # 2. Plot some raw channels before ICA
                raw.plot(n_channels=10, title='Raw EEG Before ICA', show=True)

                ################################################ ICA #######################################
                if ica:
                    raw = apply_ica(raw)

                ############################################################################################
                raw.plot(n_channels=10, title='Raw EEG After ICA', show=True)

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

def get_bci_2b(file_names_training, file_names_testing, bandpass, tmin, tmax, mode, amp_mag, baseline, ica):
    sample_rate = 250 #From BCI Dataset description
    kernels, chans = 1, 3 # There are actually 6 channels, but we only want to retain 3, as 3 are EOG
    names        = ['left', 'right']

    samples = int((tmax - (tmin)) * sample_rate)
    train_segments, train_labels = bci_2b_helper(file_names_training, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica)
    test_segments, test_labels = bci_2b_helper(file_names_testing, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica)

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

def prepare_model(X_train, X_validate, X_test, classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType, stop_threshold, input_format, model_type, freq_bins_centers, time_window_centers, n_freqs, lr, l2):
    if input_format == "timeseries":
        if model_type == "EEGNet":
            
            model = EEGNet(classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)  #try spatialdropout2d
            
        elif model_type == "Shallow":

            model = ShallowConvNet(classes, chans, samples, dropoutRate)

        elif model_type == "Deep":

            model = DeepConvNet(classes, chans, samples, dropoutRate)
        
        elif model_type == "EEGNet_Bob":
            model = EEGNet_Bob(classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType, l2_penalty=l2) 

    elif input_format == "stft": #time frequency
        print(len(time_window_centers))
        if model_type == "EEGNet":
            #Shape the 2d STFTs into 1d timeseries and feed back into EEGnet (bad idea i think)
            
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2] * X_train.shape[3], 1)
            X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2] * X_validate.shape[3], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2] * X_test.shape[3], 1)

            model = EEGNet(classes, chans, len(freq_bins_centers)*len(time_window_centers), dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)
        else:
            model = EEGNet_TF(classes, chans, len(freq_bins_centers), len(time_window_centers), dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)

    elif input_format == "wavelet":
        model = EEGNet_Wavelet3(classes, chans, n_freqs, samples, dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=2, save_best_only=True)

    print("NumParams: ", {numParams})

    if stop_threshold == 0:
        callbacks = [checkpointer]
    else:
        early_stop = EarlyStopping(
            monitor='val_loss',       # Metric to monitor
            patience=stop_threshold,              # Stop if no improvement after 30 epochs
            restore_best_weights=True  # Roll back to best weights
        )
        callbacks=[checkpointer, early_stop]

    if lr:
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",   
            factor=0.5,           
            patience=5,          
            min_lr=1e-6
        )
        callbacks.append(lr_scheduler)
        print(callbacks)
    return X_train, X_validate, X_test, model, numParams, checkpointer, callbacks

def prepare_data(X_train_raw, X_test,Y_train_raw, Y_test, sample_rate, segment_len, sample_overlap, boundary, padding, input_format, chans, samples, kernels, n_freqs, cross_validate, train_index=None, val_index=None, fold_step=None):
    freq_bins_centers, time_window_centers = None, None  
    if input_format == "stft":
        X_train, X_test, freq_bins_centers, time_window_centers = convert_stft(X_train, X_test, sample_rate, segment_len, sample_overlap, boundary, padding)
    if input_format=="wavelet":
        X_train, X_test = convert_wavelet(X_train, X_test,sample_rate, n_freqs)

    if cross_validate:
        print(f"Start fold {fold_step}")
        print("Train", train_index)
        print("Val", val_index)
        X_train = X_train_raw[train_index]
        Y_train = Y_train_raw[train_index]
        X_validate   = X_train_raw[val_index]
        Y_validate   = Y_train_raw[val_index]
        print(X_train.shape)
        print(X_validate.shape)
    else:
        # take 50/25/25 percent of the data to train/validate/test
        X_train, X_validate, Y_train, Y_validate = train_test_split(X_train_raw, Y_train_raw, test_size=0.2, stratify=Y_train_raw)

    if input_format == "timeseries":
        print(X_train.shape)
        X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

    Y_train = np_utils.to_categorical(Y_train) # One hot encoding format for probabilistic classification
    Y_validate = np_utils.to_categorical(Y_validate) # One hot encoding format for probabilistic classification
    Y_test = np_utils.to_categorical(Y_test) # One hot encoding format for probabilistic classification

    # 2) Standardize per-channel (over time & trials)
    mean  = X_train.mean(axis = (0,2), keepdims=True)
    std   = X_train.std(axis = (0,2), keepdims=True)
    X_train    = (X_train   - mean) / std
    X_validate = (X_validate- mean) / std
    X_test     = (X_test    - mean) / std

    # The ChatGPT generated wavelet model likes this format input

    if input_format=="wavelet":
        X_train = np.transpose(X_train, (0, 2, 3, 1))
        X_validate = np.transpose(X_validate, (0, 2, 3, 1)) 
        X_test = np.transpose(X_test, (0, 2, 3, 1))
    
    """if input_format=="wavelet":
        chans = 3
        X_train = X_train[:,:,:,[7,9,11]]
        X_validate = X_validate[:,:,:,[7,9,11]]
        X_test = X_test[:,:,:,[7,9,11]]"""

        # Keep only C3, C4 and CZ channels for wavelet (reccomended by wavelet paper)
        # Does not seem to work
    
    return X_train, X_test, X_validate, Y_train, Y_validate, Y_test, freq_bins_centers, time_window_centers
def plot_curves(history):

    # Accuracy curve
    plt.figure()
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss curve
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_confusion_matrix(y_pred, y_true, class_names, title="Confusion Matrix", cm=None):
    
    if cm is None: # cm param allows bypassing of this
        cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.ylabel("Predicted label")
    plt.xlabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def predict_and_visualise(X_test, Y_test, model, fittedModelHistory, names, i,logfile, sum_accuracies=0, gui_plots=True, fold_step=None):
    # load optimal model weights based on validation accuracy
    model.load_weights('/tmp/checkpoint.h5')

    #predict
    probs = model.predict(X_test)
    preds = probs.argmax(axis = -1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    sum_accuracies += acc
    best_epoch = fittedModelHistory.history['val_loss'].index(min(fittedModelHistory.history['val_loss']))

    print(model.summary())
    print("Test set accuracy: %f " % (acc))
    print("Best epoch: ", best_epoch)
    print("Average confidence of selected class: ", np.mean(probs.max(axis=1)))

    print("ABD")
    cm = confusion_matrix(preds, Y_test.argmax(axis = -1))
    class_acc = cm.diagonal() / cm.sum(axis=0)   # per-class accuracy #used to be axis=1 for recall
    print(class_acc)
    #for j in range(len(class_acc)):
     #   if math.isnan(class_acc[j]):
      #      class_acc[j] = 0
    #print(class_acc)

    # Log accuracy to file
    with open(logfile, "a") as f:
        if not fold_step ==None:
            f.write(f"Subject {i+1} Fold {fold_step} - Accuracy: {acc:.4f}. Best epoch: {best_epoch}\n")
            # in CV mode, the subject class acc gets logged later
        else:
            f.write(f"Subject {i+1} - Accuracy: {acc:.4f}. Best epoch: {best_epoch}\n")

    if gui_plots:
        plot_confusion_matrix(Y_test.argmax(axis = -1), preds, names, title = 'EEGNet-8,2')

        # XDAWN RG, Only works in time series, Also doesnt seem to work with BCI 2B
        #xdawnrg(X_train, X_test, Y_train, Y_test, chans, samples, names)

        # Show only the first 10 samples for clarity
        samples_to_plot = 10
        plot_predicted_probs(probs, samples_to_plot)

        plot_curves(fittedModelHistory.history)

        # Plot all confidences
        plot_prediction_confidence(probs)

        # plot all selected probs
        plot_all_predicted_probabilities(probs)

    return sum_accuracies, acc, class_acc, cm