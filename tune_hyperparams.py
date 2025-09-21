import util
import numpy as np
from sklearn.model_selection import StratifiedKFold
import keras_tuner as kt

# BIG 3
input_format = "timeseries"
model_type = "EEGNet_Bob"
dataset = "BCI 2b"

if dataset == "BCI 2a":
    tmin, tmax = 0.5, 2.5 # seconds before and after stimulus we want to record  # From the paper describing dataset 2A
    classes = 4
    training_files_list = [["A01T"], ["A02T"], ["A03T"],["A04T"],["A05T"], ["A06T"], ["A07T"], ["A08T"], ["A09T"]]
    #training_files_list = ["A01T", "A02T"]
    testing_files_list = [["A01E"], ["A02E"], ["A03E"],["A04E"],["A05E"], ["A06E"], ["A07E"], ["A08E"], ["A09E"]]
    #testing_files_list = ["A01E", "A02E"]
elif dataset == "BCI 2b":
    training_files_list = [['B0101T','B0102T','B0103T'],['B0201T','B0202T','B0203T'],['B0301T','B0302T','B0303T'],['B0401T','B0402T','B0403T'],['B0501T','B0502T','B0503T'],['B0601T','B0602T','B0603T'],['B0701T','B0702T','B0703T'],['B0801T','B0802T','B0803T'],['B0901T','B0902T','B0903T']]
    testing_files_list = [['B0104E','B0105E'],['B0204E','B0205E'],['B0304E','B0305E'],['B0404E','B0405E'],['B0504E','B0505E'],['B0604E','B0605E'],['B0704E','B0705E'],['B0804E','B0805E'],['B0904E','B0905E']]
    tmin, tmax = 0, 3.996 
    classes = 2
else:
    raise ValueError("INVALID DATASET")

# PREPROCESSING
bandpass = [4,40]
baseline = None
amplitude_magnification = 1000
ica = False

segment_len=128
sample_overlap=64
boundary=None
padding=True

n_freqs = 30

# MODEL DEFAULTS (may be overridden by tuner)
dropoutRate = 0.25
kernLength = 32
F1 = 8
D = 2
F2 = 16
dropoutType = 'Dropout'
stop_threshold = 100
batch_size = 32
epochs = 300   # use fewer for tuning speed
lr = False
l2 = 0.1
augment = True

# ---- Load data for ONE SUBJECT to tune hyperparams ----
subject_idx = 0  # tune on subject 1
training_files = training_files_list[subject_idx]
test_files = testing_files_list[subject_idx]

if dataset == "BCI 2a":
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2a(
        training_files, test_files,
        bandpass=bandpass, tmin=tmin, tmax=tmax,
        mode="gdf", amp_mag=amplitude_magnification,
        baseline=baseline, ica=ica,gui=False
    )
else:
    X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2b(
        training_files, test_files,
        bandpass=bandpass, tmin=tmin, tmax=tmax,
        mode="gdf", amp_mag=amplitude_magnification,
        baseline=baseline, ica=ica,gui=False
    )

# One fold only for tuning
skf = StratifiedKFold(n_splits=4, shuffle=True)
for _, val_index in skf.split(X_train_raw, Y_train_raw):
    train_index = np.setdiff1d(np.arange(len(Y_train_raw)), val_index)
    break


# ---- KerasTuner build function ----
def build_model(hp):
    dropoutRate = hp.Float("dropoutRate", 0.1, 0.7, step=0.1)
    kernLength = hp.Int("kernLength", 8, 64, step=8)
    F1 = hp.Int("F1", 4, 32, step=4)
    D = hp.Int("D", 1, 4, step=1)
    F2 = F1*D
    l2_reg = hp.Float("l2", 1e-5, 0.5, sampling="log")
    augment = hp.Boolean("augment")
    augment_chops = 5 # best = 5
    augment_probs= [0.5,0.5,0.3] #best = 0.5,0.5,0.3


    # Prepare fold data
    X_train, X_test, X_validate, Y_train, Y_validate, Y_test, freq_bins_centers, time_window_centers = util.prepare_data(
        X_train_raw, X_test_raw, Y_train_raw, Y_test_raw,
        sample_rate, segment_len, sample_overlap, boundary, padding,
        input_format, chans, samples, kernels, n_freqs,
        True, train_index, val_index, 0
    )

    if augment:
        X_train, Y_train= util.augment(X_train, Y_train, augment_chops, timeshift_prob=augment_probs[0], noise_prob=augment_probs[1], chan_dropout_prob=augment_probs[2])

    # Build model with candidate params
    X_train, X_validate, X_test, model, numParams, checkpointer, callbacks = util.prepare_model(
        X_train, X_validate, X_test,
        classes, chans, samples,
        dropoutRate, kernLength, F1, D, F2,
        dropoutType, stop_threshold,
        input_format, model_type,
        freq_bins_centers, time_window_centers,
        n_freqs, lr, l2_reg
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---- Run Bayesian tuner ----
tuner = kt.BayesianOptimization(
    build_model,
    objective="val_loss",
    max_trials=40,  # number of configs to test
    directory="tuner_results",
    project_name="eegnet_bayesian"
)

# Prepare fold data once (tuner will call build_model repeatedly)
X_train, X_test, X_validate, Y_train, Y_validate, Y_test, freq_bins_centers, time_window_centers = util.prepare_data(
    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw,
    sample_rate, segment_len, sample_overlap, boundary, padding,
    input_format, chans, samples, kernels, n_freqs,
    True, train_index, val_index, 0
)

tuner.search(
    X_train, Y_train,
    validation_data=(X_validate, Y_validate),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# ---- Results ----
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nBest Hyperparameters:")
for k, v in best_hp.values.items():
    print(f"{k}: {v}")
