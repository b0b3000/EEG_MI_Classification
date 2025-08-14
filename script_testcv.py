# %%
import util
from models import EEGNet, EEGNet_TF, EEGNet_Wavelet, ShallowConvNet, DeepConvNet

from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from tensorflow.keras import utils as np_utils
from pyriemann.utils.viz import plot_confusion_matrix

# %%
input_format = "timeseries"
model_type = "EEGNet"

training_files_list = ["A01T", "A02T", "A03T","A04T", "A05T", "A06T", "A07T", "A08T", "A09T"]
testing_files_list = ["A01E", "A02E", "A03E", "A04E", "A05E", "A06E", "A07E", "A08E", "A09E"]
# STATIC
#training_files = ["B0101T", "B0102T", "B0103T"]
#test_files = ["B0104E", "B0105E"]
tmin, tmax = 0.5, 2.5 # seconds before and after stimulus we want to record  # From the paper describing dataset
#tmin, tmax = 0, 3.996
classes = 4
dataset = "BCI 2a"

#PREPROCESSING
bandpass = [4,40]
baseline = None
amplitude_magnification = 1000 #current best = 1000

#STFT
segment_len=128
sample_overlap=64
boundary=None
padding=True

#WAVELET
n_freqs = 30

#MODEL HYPERPARAMS
dropoutRate = 0.5
kernLength = 32
F1 = 8
D = 2
F2 = 16
dropoutType = 'SpatialDropout2D'
stop_threshold = 0


# INVESTIGATE CHANGING BOUNDARY AND PADDING -> Try to get less than 17 windows -> Maybe 14
# add these as params also

with open("accuracy_log.txt", "a") as f:
    f.write(f"------------------------------------------------\n")
    f.write(f'CV\n')
    f.write(f'New Run. Dataset: {dataset}. input: {input_format}, model: {model_type}, tmin, tmax: {tmin}, {tmax} bandpass: {bandpass}, Baseline: {baseline}, Amplitude Magnification: {amplitude_magnification}, STFT -> Segment Length: {segment_len}, Sample Overlap: {sample_overlap}, Boundary: {boundary}, Padding: {padding}, Wavelet -> n_freqs: {n_freqs}, Model Hyperparams -> Dropout Rate: {dropoutRate}, Kernel Length: {kernLength}, F1: {F1}, D: {D}, F2: {F2}, Dropout Type: {dropoutType}, Stop threshold: {stop_threshold}\n')
    f.write(f"------------------------------------------------\n")

sum_accuracies = 0.0
num_subjects = len(training_files_list)

for i in range(num_subjects):
    # load that subject's *training* session (we will do 4-fold CV on this)
    training_files = [training_files_list[i]]
    test_files = [testing_files_list[i]]  # still available if you want the held-out evaluation session

    X_train_session, Y_train_session, X_test_session, Y_test_session, chans, kernels, samples, names, sample_rate = util.get_bci_2a(
        training_files, test_files,
        bandpass = bandpass,
        tmin = tmin, tmax = tmax,
        mode = "gdf",
        amp_mag= amplitude_magnification,
        baseline=baseline
    )

    # Create 4 contiguous blocks along trials (blockwise CV)
    n_trials = X_train_session.shape[0]
    blocks_X = np.array_split(X_train_session, 4, axis=0)
    blocks_Y = np.array_split(Y_train_session, 4, axis=0)

    fold_accuracies = []

    # if wavelet format needed: transpose as before (kept for completeness)
    if input_format == "wavelet":
        X_train_s = np.transpose(X_train_s, (0, 2, 3, 1))
        X_validate_s = np.transpose(X_validate_s, (0, 2, 3, 1))
        X_test_s = np.transpose(X_test_s, (0, 2, 3, 1))
        X_test_session = np.transpose(X_test_session, (0, 2, 3, 1))

    # Build a fresh model for this fold
    if input_format == "timeseries":
        if model_type == "EEGNet":
            model = EEGNet(classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)
        elif model_type == "Shallow":
            model = ShallowConvNet(classes, chans, samples, dropoutRate)
        elif model_type == "Deep":
            model = DeepConvNet(classes, chans, samples, dropoutRate)
    elif input_format == "stft":
        # build TF variant if using STFT (kept from your original block)
        X_train_s = X_train_s.reshape(X_train_s.shape[0], X_train_s.shape[1], X_train_s.shape[2]*X_train_s.shape[3], 1)
        X_validate_s = X_validate_s.reshape(X_validate_s.shape[0], X_validate_s.shape[1], X_validate_s.shape[2]*X_validate_s.shape[3], 1)
        X_test_s = X_test_s.reshape(X_test_s.shape[0], X_test_s.shape[1], X_test_s.shape[2]*X_test_s.shape[3], 1)
        model = EEGNet(classes, chans, len(freq_bins_centers)*len(time_window_centers), dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)
    else:
        # wavelet variant (name from your script)
        model = EEGNet_Wavelet3(classes, chans, n_freqs, samples, dropoutRate, kernLength, F1, D, F2, dropoutType=dropoutType)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fresh checkpoint per subject-fold so they do not interfere
    #ckpt_path = f"/tmp/checkpoint_sub{i+1}_fold{fold+1}.h5"
    checkpoint_path = '/tmp/checkpoint.h5'
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)

    if stop_threshold == 0:
        callbacks = [checkpointer]
    else:
        early_stop = EarlyStopping(monitor='val_loss', patience=stop_threshold, restore_best_weights=True)
        callbacks = [checkpointer, early_stop]

    for fold in range(4):
        # define blocks
        test_block_idx = fold
        val_block_idx = (fold + 1) % 4
        train_block_idxs = [b for b in range(4) if b not in (test_block_idx, val_block_idx)]

        # construct datasets for this fold
        X_train = np.concatenate([blocks_X[b] for b in train_block_idxs], axis=0)
        Y_train = np.concatenate([blocks_Y[b] for b in train_block_idxs], axis=0)
        X_validate = blocks_X[val_block_idx]
        Y_validate = blocks_Y[val_block_idx]
        X_test_fold = blocks_X[test_block_idx]
        Y_test_fold = blocks_Y[test_block_idx]

        # reshape to (trials, chans, samples, kernels) as in your original script
        if input_format == "timeseries":
            X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
            X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
            X_test_fold = X_test_fold.reshape(X_test_fold.shape[0], chans, samples, kernels)

        # one-hot encode labels
        Y_train_oh = np_utils.to_categorical(Y_train)
        Y_validate_oh = np_utils.to_categorical(Y_validate)
        Y_test_oh = np_utils.to_categorical(Y_test_fold)
        

        # standardize per-channel (mean/std) using the training fold only
        mean  = X_train.mean(axis = (0,2), keepdims=True)
        std   = X_train.std(axis = (0,2), keepdims=True)
        X_train_s    = (X_train   - mean) / (std + 1e-8)
        X_validate_s = (X_validate- mean) / (std + 1e-8)
        X_test_s     = (X_test_fold- mean) / (std + 1e-8)

        # fit (quiet mode as before)
        fittedModel = model.fit(
            X_train_s, Y_train_oh,
            batch_size = 16, epochs = 300,
            verbose = 0, validation_data=(X_validate_s, Y_validate_oh),
            callbacks=callbacks
        )

        # load best weights saved on validation
        model.load_weights(checkpoint_path)

        # predict on the held-out test block for this fold
        probs = model.predict(X_test_s)
        preds = probs.argmax(axis=-1)
        acc_fold = np.mean(preds == Y_test_oh.argmax(axis=-1))

        # log and collect
        with open("accuracy_log.txt", "a") as f:
            f.write(f"Subject {i+1} Fold {fold+1} - Accuracy: {acc_fold:.4f}\n")
        fold_accuracies.append(acc_fold)

        # optional: plot confusion matrix (comment out if running many folds)
        # plt.figure(); plot_confusion_matrix(preds, Y_test_oh.argmax(axis=-1), names, title=f'Sub{i+1}_Fold{fold+1}')

    # per-subject average across the 4 folds
    subj_avg = np.mean(fold_accuracies)
    with open("accuracy_log.txt", "a") as f:
        f.write(f"Subject {i+1} Avg (4-fold) - Accuracy: {subj_avg:.4f}\n")

    if input_format == "timeseries":
        X_test_session = X_test_session.reshape(X_test_session.shape[0], chans, samples, kernels)
    Y_test = np_utils.to_categorical(Y_test_session)

    probs = model.predict(X_test_session)
    preds = probs.argmax(axis=-1)
    

    acc = np.mean(preds == Y_test.argmax(axis=-1))
    # Log accuracy to file
    with open("accuracy_log.txt", "a") as f:
        f.write(f"Subject {i+1} - TEST Accuracy: {acc:.4f}\n")
    sum_accuracies += acc
    # ADD A TEST SET EVALUATION - > PROMISING

avg_acc = sum_accuracies/9
print("AVERAGE ACCURACY: ", avg_acc)
with open("accuracy_log.txt", "a") as f:
    f.write(f"Total Accuracy: {avg_acc:.4f}\n")

# overall average across subjects
overall_avg = sum_accuracies / num_subjects
print("AVERAGE ACCURACY (subject-specific 4-fold mean across subjects):", overall_avg)
with open("accuracy_log.txt", "a") as f:
    f.write(f"Overall Average (subjects): {overall_avg:.4f}\n")

