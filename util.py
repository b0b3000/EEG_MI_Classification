from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np
import mne
from mne import io
from mne.datasets import sample
from models import EEGNet, EEGNet_Modified, ShallowConvNet, DeepConvNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import stft
import pywt
import os

FIGURE_DIR = "/Users/bobbeashel/Desktop/CITS4010/Project/figures/"
DATASET_LOCATION = "/Users/bobbeashel/Desktop/CITS4010/Project/data/"

def savefig_unique(fig, filepath, fig_obj=True):
    '''
    This function ensures that all generated figures are:
    1) saved as image files in the figure directory
    2) saved with a unique name, so no figures are overwritten
    '''
    counter = 0
    name, ext = filepath.rsplit(".", 1)
    name = FIGURE_DIR + name
    
    #first attempt
    unique_path = FIGURE_DIR + filepath

    #keep adding a number until free -> so no overwrite
    while os.path.exists(unique_path):
        counter += 1
        unique_path = f"{name}{counter}.{ext}"

    if fig_obj:
        fig.savefig(unique_path)
    else:
        #fig.figure.savefig(unique_path)
        fig.savefig(unique_path)

    print(f"[INFO] Saved: {unique_path}")

def augment_trial(x, timeshift_prob, noise_prob, chan_dropout_prob):
    """
    Apply random augmentation transformations to a single EEG trial.

    This function probabilistically applies up to three augmentations:
    1. Time-shifting along the sample axis.
    2. Gaussian noise injection.
    3. Random channel dropout (zeroing selected channels)."""
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
    
    if np.random.rand() < timeshift_prob:
        x = time_shift(x, 20) # max shift = 20
    if np.random.rand() < noise_prob:
        x = add_gaussian_noise(x, 1e-6) #gaussian param
    if np.random.rand() < chan_dropout_prob:
        x = channel_dropout(x, 2) # dropout 2 channels randomly
    return x

#This function applies the augmentation step in pre-processing. It draws on augment_trial.
def augment(X_train, Y_train, n_segments, timeshift_prob=0.5, noise_prob=0.5, chan_dropout_prob=0.3):
    '''For each original trial, this function:
    - Divides the trial into n_segments equal length temporal chunks.
    - For each segment, samples a random trial from the same class.
    - Replaces the corresponding segment with an augmented version
      (via time shift, noise, and/or dropout).
    - Appends the resulting recombined trial to the training set.'''
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

def visualise_sample_wavelet(samples_per_trial, freqs, X_train_converted, sample_rate):
    '''
    This function is for visualising a single channel of a single trial
    after wavelet transform, for a visual comparison
    '''
    times = np.arange(samples_per_trial) / sample_rate #500/250 = 2
    plt.figure(figsize=(10, 6))
    plt.contourf(times, freqs, X_train_converted[0][0], levels=100, cmap='viridis')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time-Frequency Representation (Wavelet Transform) Trial 1 Channel 1')
    plt.colorbar(label='Power')
    plt.ylim(0,60)
    plt.tight_layout()
    savefig_unique(plt, "wavelet_transform.png")

#Converts inputted signal arrays into wavelet spectographs
def convert_wavelet(X_train, X_test, sample_rate, num_frequencies):
    samples_per_trial = X_train.shape[2]
    fmin = 2
    fmax = sample_rate/2
    wavelet = "cmor3-3"

    plot_single_channel(X_train[0,0], sample_rate, "training", "before_wavelet.png")

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

    visualise_sample_wavelet(samples_per_trial, freqs, X_train_converted, sample_rate)

    return X_train_converted, X_test_converted

def plot_all_predicted_probabilities(probs, title=None, class_names=None):
    """
    Plot the distribution of maximum predicted probabilities for each instance,
    including both individual dots (stripplot) and summary (boxplot).
    
    Args:
        probs: array of shape (n_samples, n_classes)
    """
    max_probs = probs.max(axis=1)  # get top-1 probability for each sample

    plt.figure(figsize=(12, 5))
    
    # Create both stripplot and boxplot on the same axis
    sns.stripplot(x=max_probs, orient='h', jitter=0.2, alpha=0.5, color='blue', label='Individual Samples')
    sns.boxplot(x=max_probs, orient='h', color='lightgray', width=0.3, fliersize=0, linewidth=1)

    plt.xlabel('Top-1 Predicted Probability')
    plt.title(f'{title} Predicted Probability Distribution')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend()
    savefig_unique(plt, "all_prob_distributions.png")
    plt.close()


# Plots the first trial and channel for visualisation
def plot_single_channel(x, sample_rate, dataset, title):
    t = np.arange(len(x)) / sample_rate  # time axis in seconds
    print('ABCDE ', x.shape)
    plt.figure(figsize=(10, 4))
    plt.plot(t, x, color="b")
    plt.title(f"Raw Timeseries (Channel 0, Trial 0) - {dataset}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    savefig_unique(plt, title)

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

    savefig_unique(plt, "confusion_xdawnrg.png")

def convert_stft(X_train, X_test, sample_rate, segment_len=64, sample_overlap=32, boundary="zeros", padding=True):
    def compute_stft(X,dataset):
        plot_single_channel(X[0,0], sample_rate, dataset, "before_stft.png")
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
    '''
    This function is for visualising a single channel of a single trial
    after wavelet transform, for a visual comparison
    '''
    sample_stft = np.abs(sample_stft)  
    print("Each sample + channel has", len(freqs), "frequency bins with", len(times), "windows each.")

    plt.figure(figsize=(12, 5))
    pcm = plt.pcolormesh(times, freqs, sample_stft, shading='nearest') #gourad
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('STFT Magnitude — Trial 1, Channel 1 ' + dataset)

    # time grid
    for t in times:
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=0.3)
    # frequency grid
    for f in freqs:
        plt.axhline(y=f, color='gray', linestyle='--', linewidth=0.3)
    
    plt.ylim(0, 60)

    # Add colorbar
    plt.colorbar(pcm, label='Magnitude')
    plt.tight_layout()
    savefig_unique(plt, "stft.png")
    plt.close()


def plot_predicted_probs(probs, num_samples_to_plot, title="Predicted Probabilities"):
    '''
    Visualises class probability distributions for a subset of prediction samples as stacked bar charts.

    Each bar represents one sample, subdivided into colored segments corresponding to class probabilities.
    The colors are ranked by probability magnitude (red = most likely, blue = 2nd, green = 3rd, yellow = 4th).
    Probabilities for each sample are sorted in descending order before plotting.

    '''
    subset = probs[:num_samples_to_plot]

    labels = [f'Sample {i}' for i in range(num_samples_to_plot)]
    classes = [f'Class {i}' for i in range(probs.shape[1])]
    
    # rank-based colors. Red will always represent the highest probability, and so on.
    colors = ["red", "blue", "green", "yellow"]  

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
    plt.title(f'{title} (First {num_samples_to_plot} Samples)')

    # Legend should show "Most likely", "2nd", etc.
    # the same order remains
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
    savefig_unique(plt, "predicted_probs.png")
    plt.close()


def plot_prediction_confidence(probs, title="Prediction Confidences", k=1.2):

    """
    Plots a strip plot of Top1/Top2 confidence ratios for all samples, capped at 5.
    
    Parameters:
    - probs: 2D numpy array of shape (num_samples, num_classes)
    """
    sorted_probs = -np.sort(-probs, axis=1)
    top1 = sorted_probs[:, 0]
    top2 = sorted_probs[:, 1]

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        confidence_ratio = np.where(top2 != 0, top1 / top2, np.inf)

    # Cap extreme values at 5 for visualization clarity
    confidence_ratio = np.clip(confidence_ratio, a_min=1, a_max=5)

    # Seaborn style
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 5))
    # Highlight region from 1 to k
    plt.axvspan(1, k, color='red', alpha=0.2, label=f'Uncertain Predictions Region (Confidence < {k})')
    sns.histplot(confidence_ratio, kde=True, bins=30, color='skyblue')
    plt.legend()
    plt.title(f"{title} Confidence Ratios")
    plt.xlim(1, 5)
    plt.xticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5+"])
    plt.xlabel("Confidence Ratio")
    plt.ylabel("Frequency")

    savefig_unique(plt, "top_confidence_distribution.png")
    plt.close()


def exponential_moving_standardize(train_segments, test_segments, decay=0.999, init_block_size=1000):
    '''
    Apply exponential moving standardization to EEG data.

    This function performs online normalisation by updating mean and variance
    estimates over time using exponential decay. It first standardises all
    training segments, then applies the same normalisation state
    (mean and variance) to the test segments.

    Returns:
    standardized_train : array
        Array of shape (n_trials, n_channels, n_samples) containing standardized training data.
    standardized_test : array
        Array of shape (n_trials, n_channels, n_samples) containing standardized test data.
    '''
   
    train_cont = np.concatenate(train_segments, axis=1)  # shape: (n_channels, total_samples)
    test_cont  = np.concatenate(test_segments, axis=1)

    n_channels, n_total_train_samples = train_cont.shape
    _, n_total_test_samples = test_cont.shape

    init_mean = np.mean(train_cont[:, :init_block_size], axis=1, keepdims=True)
    init_var  = np.var(train_cont[:, :init_block_size], axis=1, keepdims=True)

    train_out = np.zeros_like(train_cont)
    test_out  = np.zeros_like(test_cont)

    mean_t = init_mean.copy()
    var_t  = init_var.copy()

    for t in range(n_total_train_samples):
        x_t = train_cont[:, t:t+1] 
        mean_t = (1 - decay) * x_t + decay * mean_t
        var_t  = (1 - decay) * (x_t - mean_t) ** 2 + decay * var_t
        train_out[:, t:t+1] = (x_t - mean_t) / np.sqrt(var_t + 1e-8)

    for t in range(n_total_test_samples):
        x_t = test_cont[:, t:t+1]
        mean_t = (1 - decay) * x_t + decay * mean_t
        var_t  = (1 - decay) * (x_t - mean_t) ** 2 + decay * var_t
        test_out[:, t:t+1] = (x_t - mean_t) / np.sqrt(var_t + 1e-8)

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

# This function applies the independent component analysis (ICA) transformation to inputted raw signal data.
# This acts as a form of artifact removal in pre-processing
def apply_ica(raw, gui):
    raw.set_channel_types({ch: 'eog' for ch in raw.ch_names[-3:]})

    ica = mne.preprocessing.ICA(n_components=20)
    ica.fit(raw)
    
    if gui:
        ica.plot_sources(raw, show=False)  
        savefig_unique(plt, "ica_sources_before.png")
        plt.close()


    # Detect components correlated with EOG
    ica.exclude = ica.find_bads_eog(raw)[0]

    # Apply ICA
    raw = ica.apply(raw)

    #Plot ICA sources for visualisation
    if gui:
        ica.plot_sources(raw, show=False)  
        savefig_unique(plt, "ica_sources_after.png") 
        plt.close()


    # Now drop EOG before epochs
    raw.pick_types(eeg=True)

    #Plot ICA overlay
    if gui:
        ica.plot_overlay(raw, exclude=ica.exclude, picks='eeg', show=False)
        savefig_unique(plt, "ica_overlay.png")
        plt.close()


    print(f"Components to remove: {ica.exclude}")

    return raw


def bci_2a_helper(file_names, tmin, tmax, chans,bandpass, mode, amp_mag, baseline, ica, gui):
    """
    Helper function to load and preprocess EEG data from the BCI Competition IV 2a dataset.

    This function loads .gdf EEG files and their corresponding .mat label files,
    applies specified pre-processing, and epoching to create fixed-length processed trials with associated class labels.
    Args:
    file_names : List of subject/session file names (without extensions) to load.
    tmin, tmax : Start and end times (in seconds) for epoch extraction relative to cue onset.
    chans : Number of EEG channels to retain
    bandpass : Frequency band (low_cutoff, high_cutoff) for band-pass filtering.
    mode : Dataset mode, e.g., "gdf" for loading from .gdf files.
    amp_mag : Amplitude scaling factor to adjust signal magnitude.
    baseline : Baseline correction period for MNE epoching.
    ica : Whether to apply Independent Component Analysis (ICA) for artifact removal.
    gui : If True, display diagnostic plots for visual inspection.
    """

    if file_names:
        all_segments = []
        all_labels = []
        if mode == "gdf":
            directory = DATASET_LOCATION + "BCICIV_2a_gdf/" #.gdf files locations
            for i, file_name in enumerate(file_names):
                data_path = directory + file_name + ".gdf"
                labels_path = directory + "true_labels/" + file_name + ".mat"
                raw = mne.io.read_raw_gdf(data_path, preload=True, verbose=0)

                ################################################ BANDPASS ####################################4
                raw.filter(bandpass[0],bandpass[1], fir_design='firwin', skip_by_annotation='edge', verbose=0)
                #raw.filter(2, None, method='iir') 

                ################################################## ICA ###################################
                raw_before_ica = raw.copy()
                if ica:
                    raw = apply_ica(raw,gui)

                # visual only
                if gui:
                    raw_before_ica.plot(n_channels=5, duration=5, title="Before ICA")
                    raw.plot(n_channels=5, duration=5, title="After ICA")

                    fig1 = raw_before_ica.plot_psd(fmax=50, show=False)
                    fig1.suptitle("PSD Before ICA")

                    fig2 = raw.plot_psd(fmax=50, show=False)
                    fig2.suptitle("PSD After ICA")

                    savefig_unique(fig1, "before_ica.png",  False)
                    savefig_unique(fig2, "after_ica.png",False)

                    #fig1.show()
                    #fig2.show()
                    
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

def get_bci_2a(file_names_training, file_names_testing, bandpass, tmin, tmax, mode, amp_mag, baseline, ica, gui):
    """
    Load, preprocess, and standardize training and testing data for BCI Competition IV 2a.

    This function orchestrates the full preprocessing pipeline:"""
    sample_rate = 250 #From BCI Dataset description
    kernels, chans = 1, 22 # There are actually 25 channels, but we only want to retain 22, as 3 are EOG
    names        = ['left', 'right', 'foot', 'tongue']

    samples = int((tmax - (tmin)) * sample_rate)
    train_segments, train_labels = bci_2a_helper(file_names_training, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica, gui)
    test_segments, test_labels = bci_2a_helper(file_names_testing, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica, gui)

    train_segments, test_segments = exponential_moving_standardize(train_segments, test_segments)

    
    
    return(train_segments, train_labels, test_segments, test_labels, chans, kernels, samples, names, sample_rate)

def bci_2b_helper(file_names, tmin, tmax, chans,bandpass, mode, amp_mag, baseline, ica, gui):
    """
    Helper function to load and preprocess EEG data from the BCI Competition IV 2b dataset.

    This function loads .gdf EEG files and their corresponding .mat label files,
    applies specified pre-processing, and epoching to create fixed-length processed trials with associated class labels.
    Args:
    file_names : List of subject/session file names (without extensions) to load.
    tmin, tmax : Start and end times (in seconds) for epoch extraction relative to cue onset.
    chans : Number of EEG channels to retain
    bandpass : Frequency band (low_cutoff, high_cutoff) for band-pass filtering.
    mode : Dataset mode, e.g., "gdf" for loading from .gdf files.
    amp_mag : Amplitude scaling factor to adjust signal magnitude.
    baseline : Baseline correction period for MNE epoching.
    ica : Whether to apply Independent Component Analysis (ICA) for artifact removal.
    gui : If True, display diagnostic plots for visual inspection.
    """
    
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

                ################################################## ICA ###################################
                raw_before_ica = raw.copy()
                if ica:
                    raw = apply_ica(raw, gui)
                    
                    # visual only
                    if gui:
                        raw_before_ica.plot(n_channels=5, duration=5, title="Before ICA")
                        raw.plot(n_channels=5, duration=5, title="After ICA")

                        fig1 = raw_before_ica.plot_psd(fmax=50, show=False)
                        fig1.suptitle("PSD Before ICA")

                        fig2 = raw.plot_psd(fmax=50, show=False)
                        fig2.suptitle("PSD After ICA")

                        #fig1.show()
                        #fig2.show()
                        savefig_unique(fig1, "PSD_before_ica.png",  False)
                        plt.close()

                        savefig_unique(fig2,"PSD_after_ica.png",  False)
                        plt.close()

                ####################################################################################
                

                events, _ = mne.events_from_annotations(raw, event_id = {
                    '769': 1,   # left hand
                    '770': 2,   # right hand
                    '783': 3   #unknown (eval sets)
                }, verbose = 0)

                #Epoch data into windowed trials
                epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=0)

                #Get the signal data from the EEG channels of epoch. omit EOG signals
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

def get_bci_2b(file_names_training, file_names_testing, bandpass, tmin, tmax, mode, amp_mag, baseline, ica, gui):
    """
    Load, preprocess, and standardize training and testing data for BCI Competition IV 2b.

    This function orchestrates the full preprocessing pipeline:
    """
    sample_rate = 250 #From BCI Dataset description
    kernels, chans = 1, 3 # There are actually 6 channels, but we only want to retain 3, as 3 are EOG
    names        = ['left', 'right']

    samples = int((tmax - (tmin)) * sample_rate)
    train_segments, train_labels = bci_2b_helper(file_names_training, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica, gui)
    test_segments, test_labels = bci_2b_helper(file_names_testing, tmin, tmax, chans, bandpass, mode, amp_mag, baseline, ica, gui)

    train_segments, test_segments = exponential_moving_standardize(train_segments, test_segments)

    return(train_segments, train_labels, test_segments, test_labels, chans, kernels, samples, names, sample_rate)

def prepare_model(X_train, X_validate, X_test, classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType, stop_threshold, input_format, model_type, freq_bins_centers, time_window_centers, n_freqs, lr, l2):
    """
    Prepare, configure, and compile a model for EEG classification. Dynamically builds and compiles an appropriate neural network model
    based on the provided input format and model type. It also sets up callbacks such as checkpointing, early stopping, and learning rate scheduling.

    Parameters
    ----------
    X_train, X_validate, X_test : Training, validation, and test data arrays.
    classes : Number of output classes (e.g., 4 for BCI-IV 2a).
    chans : Number of EEG channels.
    samples : Number of samples (time-points) per trial.
    dropoutRate : Dropout probability for regularization.
    kernLength : Length of temporal convolution kernels.
    F1, D, F2 : EEGNet hyperparameters controlling filter counts and depth multiplier.
    dropoutType : Type of dropout ('Dropout' or 'SpatialDropout2D').
    stop_threshold : Early stopping patience (in epochs); 0 disables early stopping.
    input_format : Input data type — one of {'timeseries', 'stft', 'wavelet'}.
    model_type : Model architecture to instantiate — one of {'EEGNet', 'Shallow', 'Deep', 'EEGNet_Modified'}.
    freq_bins_centers : Frequency bin centers (used for STFT-based models).
    time_window_centers : Time window centers (used for STFT-based models).
    n_freqs : Number of frequency components (for wavelet input).
    lr : Whether to include a learning rate scheduler callback.
    l2 : L2 regularisation penalty (used in modified EEGNet).

    Returns
    -------
    X_train, X_validate, X_test : Possibly reshaped datasets depending on input format.
    model : Compiled model ready for training.
    numParams : Total number of trainable model parameters.
    checkpointer : Callback for saving best-performing model weights.
    callbacks : List of callbacks used during training.
    """
    
    #If timeseries input is specified, create the model based on the specified model.
    if input_format == "timeseries":
        if model_type == "EEGNet":
            
            model = EEGNet(classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)  #try spatialdropout2d
            
        elif model_type == "Shallow":

            model = ShallowConvNet(classes, chans, samples, dropoutRate)

        elif model_type == "Deep":

            model = DeepConvNet(classes, chans, samples, dropoutRate)
        
        elif model_type == "EEGNet_Modified":
            model = EEGNet_Modified(classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType, l2_penalty=l2) 

    #If stft input is specified, create the model based on the specified model.
    elif input_format == "stft": #time frequency
        print(len(time_window_centers))
        if model_type == "EEGNet":

            #Shape the 2d STFTs into 1d timeseries and feed back into EEGnet (does not work well - would not reccomend doing this)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2] * X_train.shape[3], 1)
            X_validate = X_validate.reshape(X_validate.shape[0], X_validate.shape[1], X_validate.shape[2] * X_validate.shape[3], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2] * X_test.shape[3], 1)

            model = EEGNet(classes, chans, len(freq_bins_centers)*len(time_window_centers), dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)
        #elif model_type == "EEGNet_STFT":
            #This option is disabled for now, as we abandoned the EEGNet_STFT variant we created.
            
    #elif input_format == "wavelet":
        #This option is disabled for now, as we also  abandoned the EEGNet_Wavelet variant we created.

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    numParams    = model.count_params()   
    print("NumParams: ", {numParams}) 

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=2, save_best_only=True)

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
    """
    Prepare train/val/test datasets for EEG classification.

    Steps:
    1) Optional transform of inputs into STFT or wavelet domains.
    2) Split into train/validation (CV indices or stratified split).
    3) Reshape to expected (n, chans, samples, kernels) for time-series.
    4) One-hot encode labels.
    5) Per-channel standardization using train-set stats.
    6) Optional axis transpose for wavelet models.

    Params:
    X_train_raw : Raw training trials; shape depends on input_format.
    X_test : Raw test trials matching X_train_raw structure.
    Y_train_raw, Integer class labels (not one-hot).
    sample_rate : Sampling frequency (Hz).
    segment_len, sample_overlap, boundary, padding : STFT parameters passed to `convert_stft`.
    input_format : One of {"timeseries", "stft", "wavelet"}.
    chans, samples, kernels : Expected dims for time-series reshape.
    n_freqs : Number of frequencies for wavelet transform.
    cross_validate : If True, use provided indices for train/val split.
    train_index, val_index : Indices used when cross_validate=True.
    fold_step : Fold counter (printed for logging).

    Returns
    X_train, X_test, X_validate : Prepared inputs ready for model consumption.
    Y_train, Y_validate, Y_test : One-hot encoded labels.
    freq_bins_centers, time_window_centers : data for STFT grids (None for other formats).
    """
    freq_bins_centers, time_window_centers = None, None  
    if input_format == "stft":
        X_train_raw, X_test, freq_bins_centers, time_window_centers = convert_stft(X_train_raw, X_test, sample_rate, segment_len, sample_overlap, boundary, padding)
    if input_format=="wavelet":
        X_train_raw, X_test = convert_wavelet(X_train_raw, X_test,sample_rate, n_freqs)

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

    # Standardize per-channel (over time & trials)
    mean  = X_train.mean(axis = (0,2), keepdims=True)
    std   = X_train.std(axis = (0,2), keepdims=True)
    X_train    = (X_train   - mean) / std
    X_validate = (X_validate- mean) / std
    X_test     = (X_test    - mean) / std

    if input_format=="wavelet":
        X_train = np.transpose(X_train, (0, 2, 3, 1))
        X_validate = np.transpose(X_validate, (0, 2, 3, 1)) 
        X_test = np.transpose(X_test, (0, 2, 3, 1))
    
    return X_train, X_test, X_validate, Y_train, Y_validate, Y_test, freq_bins_centers, time_window_centers

#Given training history, plots train and validation accuracy/loss curves
def plot_curves(history, title="Accuracy and Loss Curves"):


    # If multiple histories given, average them and display an aggregate 
    # Do not recommend doing this. the resulting plot doesnt make that much sense
    # if isinstance(history, list):
        #history = average_histories(history)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Accuracy curve
    ax1.plot(history['accuracy'], label='Train Acc')
    ax1.plot(history['val_accuracy'], label='Val Acc')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training & Validation Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss curve
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training & Validation Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)

    savefig_unique(plt, "accuracy_loss_curves.png")
    plt.close()

     

#Given the y predictions and true y labels, plot a confusion matrix. 
#Can also bypass the construction  by inputting directly the cm object if it is already made
def plot_confusion_matrix(y_pred, y_true, class_names, title="Confusion Matrix", cm=None):
    
    if cm is None: # cm param allows bypassing of this
        cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.ylabel("Predicted label")
    plt.xlabel("True label")
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    savefig_unique(plt, "confusion_matrix.png")
    plt.close()

     



def predict_and_visualise(X_test, Y_test, model, fittedModelHistory, names, i,logfile, sum_accuracies=0, gui_plots=True, fold_step=None):

    """
    Evaluate a trained EEG model on the test set and optionally visualize results.

    Loads the best saved weights, performs prediction, computes accuracy metrics,
    logs results to file, and if enabled displays confusion matrix, probability
    plots, learning curves, etc.

    Parameters
    ----------
    X_test : Test inputs of shape (n_trials, n_channels, n_samples, 1).
    Y_test : One-hot encoded true labels for the test set.
    model : Trained model instance whose weights are re-loaded from checkpoint.
    fittedModelHistory : History object returned by `model.fit()`, used to extract best epoch.
    names : Class names for axis labels in plots (e.g., ["left", "right", "foot", "tongue"]).
    i : Subject index (used in printouts/log file).
    logfile : Path to text file where accuracies are appended.
    sum_accuracies : Running sum of accuracies across folds/subjects; default = 0.
    gui_plots : If True, produce diagnostic plots.
    fold_step : Current fold index when doing cross-validation.

    Returns
    -------
    sum_accuracies : Updated accumulated accuracy total.
    acc : Test accuracy for this run.
    class_acc : Per class accuracy derived from confusion matrix.
    cm : Confusion matrix 
    """

    # load optimal model weights based on validation accuracy
    model.load_weights('/tmp/checkpoint.h5')

    #predict
    probs = model.predict(X_test)
    preds = probs.argmax(axis = -1)
    y_true = Y_test.argmax(axis=-1)
    acc = np.mean(preds == Y_test.argmax(axis=-1))
    sum_accuracies += acc
    best_epoch = fittedModelHistory.history['val_loss'].index(min(fittedModelHistory.history['val_loss']))

    #model outputs
    print(model.summary())
    print("Test set accuracy: %f " % (acc))
    print("Best epoch: ", best_epoch)
    print("Average confidence of selected class: ", np.mean(probs.max(axis=1)))

    cm = confusion_matrix(preds, Y_test.argmax(axis = -1))
    class_acc = cm.diagonal() / cm.sum(axis=0)   # per-class accuracy #used to be axis=1 for recall
    print(class_acc)

    # Log accuracy to file
    with open(logfile, "a") as f:
        if not fold_step ==None:
            f.write(f"Subject {i+1} Fold {fold_step} - Accuracy: {acc:.4f}. Best epoch: {best_epoch}\n")
            # in CV mode, the subject class acc gets logged later
        else:
            f.write(f"Subject {i+1} - Accuracy: {acc:.4f}. Best epoch: {best_epoch}\n")

    if gui_plots:
        plot_confusion_matrix(Y_test.argmax(axis = -1), preds, names, title = f"Subject {i+1} Fold {fold_step}")

        # Show only the first 10 samples for clarity
        samples_to_plot = 10
        plot_predicted_probs(probs, samples_to_plot, title = f"Subject {i+1} Fold {fold_step}")

        #plot validation accuracy curves for train/val
        plot_curves(fittedModelHistory.history, title = f"Subject {i+1} Fold {fold_step}")

        # Plot all confidences
        plot_prediction_confidence(probs, title = f"Subject {i+1} Fold {fold_step}")

        # plot all selected probs
        plot_all_predicted_probabilities(probs, title = f"Subject {i+1} Fold {fold_step}")

    return sum_accuracies, acc, class_acc, cm, y_true, preds, probs, fittedModelHistory.history

def average_histories(histories):
    """
    Takes a list of Keras history.history dicts and returns
    a single averaged history dict (element-wise mean across epochs).
    """
    # Determine max length
    max_epochs = max(len(h['val_loss']) for h in histories)
    keys = histories[0].keys()
    averaged = {}

    for key in keys:
        # Initialise array (num_histories x max_epochs)
        arr = np.zeros((len(histories), max_epochs))
        for i, h in enumerate(histories):
            if key in h:
                arr[i, :len(h[key])] = h[key]
        averaged[key] = arr.mean(axis=0)

    return averaged

def aggregate_plot(all_true, all_preds, all_probs, all_histories):
    all_true_flat = np.concatenate(all_true)
    all_preds_flat = np.concatenate(all_preds)
    all_probs_flat = np.concatenate(all_probs)
    avg_hist = average_histories(all_histories)

    plot_all_predicted_probabilities(all_probs_flat, title="Aggregate Prediction Confidence")

    plot_curves(avg_hist, title="Aggregate Learning Curves")

