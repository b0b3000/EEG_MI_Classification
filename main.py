
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
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

AMPLITUDE_MAGNIFICATION = 10000 #current best = 1000
DATASET_LOCATION = "/Volumes/My Passport/Honours_Datasets/"

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
    X = epochs.get_data() * AMPLITUDE_MAGNIFICATION 

    return(X, labels, chans, kernels, samples, names)
     
def bci_2a_helper(file_names, directory, tmin, tmax, chans):
    all_segments = []
    all_labels = []
    for i, file_name in enumerate(file_names):
        file_path = directory + file_name + ".gdf"
        file_path_labels = directory + "true_labels_2a/" + file_name + ".mat"
        raw = mne.io.read_raw_gdf(file_path, preload=True)

        ################################################ BANDPASS ####################################
        
        #raw.filter(4, 40, fir_design='firwin', skip_by_annotation='edge')
        raw.filter(2, None, method='iir') 

        ###############################################################################################

        # Extract event markers
        events, _ = mne.events_from_annotations(raw)
        print(events)

        # Filter ONLY relevant events
        motor_event_ids = [7, 8, 9, 10]  # corresponds to 'left', 'right', 'foot', 'tongue'
        mi_events = np.array([e for e in events if e[2] in motor_event_ids])
        epochs = mne.Epochs(raw, mi_events, motor_event_ids, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True)

        all_segments.append(epochs.get_data()[:,:chans,:-1])
        print(all_segments[i].shape)

        labels_raw = loadmat(file_path_labels)
        labels_raw = labels_raw["classlabel"].reshape(-1) - 1
        all_labels.append(labels_raw)

    segments= np.concatenate(all_segments, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # scale due to scaling sensitivity in deep learning
    segments = segments * AMPLITUDE_MAGNIFICATION

    # Np array format
    segments = np.array(segments)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels) # One hot encoding format for probabilistic classification

    return segments, labels

def get_bci_2a():
    sample_rate = 250 #From BCI Dataset description
    kernels, relevant_channels = 1, 22 # There are actually 25 channels, but we only want to retain 22, as 3 are EOG
    tmin, tmax = -0.5, 2 # seconds before and after stimulus we want to record 
    names        = ['left', 'right', 'foot', 'tongue']
    directory = DATASET_LOCATION + "BCICIV_2a_gdf/"
    file_names_training = ["A01T", "A02T", "A03T", "A05T", "A06T", "A07T", "A08T", "A09T"]
    #file_names_testing = ["A01E", "A02E", "A03E"]

    chans = relevant_channels

    samples = int((tmax - (tmin)) * sample_rate)

    train_segments, train_labels = bci_2a_helper(file_names_training, directory, tmin, tmax, chans)
    #test_segments, test_labels = bci_2a_helper(file_names_testing, directory, tmin, tmax, chans) # try to implement

    return(train_segments, train_labels, chans, kernels, samples, names)

X, Y, chans, kernels, samples, names = get_bci_2a()

# take 50/25/25 percent of the data to train/validate/test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.5, stratify=Y)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp)

X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# 2) Standardize per-channel (over time & trials)
mean  = X_train.mean(axis = (0,2), keepdims=True)
std   = X_train.std(axis = (0,2), keepdims=True)
X_train    = (X_train   - mean) / std
X_validate = (X_validate- mean) / std
X_test     = (X_test    - mean) / std

print('X_train shape:', X_train.shape)
print('y train shape', Y_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# count number of parameters in the model
numParams    = model.count_params()    

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0:1, 1:1, 2:1, 3:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer])

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

print(model.summary())
probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in 
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg     = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train      = X_train.reshape(X_train.shape[0], chans, samples)
X_test       = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, Y_train.argmax(axis = -1))
preds_rg     = clf.predict(X_test)

# Printing the results
acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
print("Classification accuracy: %f " % (acc2))

# plot the confusion matrices for both classifiers

plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')

plt.figure(1)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'xDAWN + RG')

plt.show()

