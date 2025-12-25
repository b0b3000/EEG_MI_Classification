import util
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import datetime

# Title of the trial for reference in logs
title = "EEGNet_Modified"

# Specify the input format, model type and dataset.

input_format = "timeseries"
model_type = "EEGNet_Modified"
dataset = "BCI 2a"

# Set some variables based on the dataset. Currently only supports BCI Competition Datset IV 2a & 2b
if dataset == "BCI 2a":
    tmin, tmax = 0.5, 2.5 # seconds before and after stimulus we want to record  # From the paper describing dataset 2A
    classes = 4
    training_files_list = [["A01T"], ["A02T"], ["A03T"],["A04T"],["A05T"], ["A06T"], ["A07T"], ["A08T"], ["A09T"]]
    testing_files_list = [["A01E"], ["A02E"], ["A03E"],["A04E"],["A05E"], ["A06E"], ["A07E"], ["A08E"], ["A09E"]]
elif dataset == "BCI 2b":
    training_files_list = [['B0101T','B0102T','B0103T'],['B0201T','B0202T','B0203T'],['B0301T','B0302T','B0303T'],['B0401T','B0402T','B0403T'],['B0501T','B0502T','B0503T'],['B0601T','B0602T','B0603T'],['B0701T','B0702T','B0703T'],['B0801T','B0802T','B0803T'],['B0901T','B0902T','B0903T']]
    testing_files_list = [['B0104E','B0105E'],['B0204E','B0205E'],['B0304E','B0305E'],['B0404E','B0405E'],['B0504E','B0505E'],['B0604E','B0605E'],['B0704E','B0705E'],['B0804E','B0805E'],['B0904E','B0905E']]
    tmin, tmax = 0, 3.996 
    classes = 2
else:
    print("INVALID DATASET")
    
#PRE-PROCESSING PARAMETERS
bandpass = [4,40] # best: [4,40]
baseline = None #best: None
amplitude_magnification = 1000 #best: 1000
ica = False # Best: True

# STFT 
segment_len=128
sample_overlap=64
boundary=None
padding=True

# WAVELET
n_freqs = 30


#MODEL HYPERPARAMS
dropoutRate = 0.5 # Best 0.4
kernLength = 32 # Best 32
F1 = 4 # Best 4
D = 4 # Best 4
F2 = 16 # F1 * D works best. (Best 16)
dropoutType = 'Dropout' # Best: 'Dropout'
stop_threshold = 150 # Best: 150
batch_size = 64 # Best: 64
epochs = 1000 # Best: 1000
lr = False # Learning rate scheduler: best = False
l2 = 0.1 # L2 regulariser (None if off), Best: 0.1

augment = True # Best: True. False if off
augment_chops = 5 # Best: 5
augment_probs= [0.5,0.5,0.3] # Best: 0.5,0.5,0.3

# False if you dont want plots produced. True if you do.
gui=False 

#Same or cross subject
same_subject = True
#Number of CV folds
folds=4

if same_subject:
    logfile = "logs/samesubject_cv_log.txt"
else:
    logfile = "logs/crosssubject_log.txt"

# Non tunable definitions.
sum_accuracies = 0
sum_class_acc = np.zeros(classes)
all_subjects_cm = []
subjects = len(training_files_list)

# Write all the parameters (listed above) to a log file. This then records all the accuracies and other stats.
# This is used to perform ablation studies, determine which models are better, etc.
with open(logfile, "a") as f:
    f.write("\n------------------------------------------------\n")
    if same_subject:
        f.write(f"SAME SUBJECT {title}\n")
    else:
        f.write(f"CROSS SUBJECT {title}\n")
    f.write(f"{datetime.datetime.now()}\n")
    f.write(f"{dataset}\n")
    f.write(f"Input: {input_format}, Model: {model_type}\n")
    f.write(f"Time Window: {tmin}-{tmax}\n")
    f.write(f"Bandpass: {bandpass}, ICA: {ica}\n")
    f.write(f"Baseline: {baseline}, Amplitude Magnification: {amplitude_magnification}\n")
    f.write(f"STFT -> Segment Length: {segment_len}, Overlap: {sample_overlap}, "
            f"Boundary: {boundary}, Padding: {padding}\n")
    f.write(f"Wavelet -> n_freqs: {n_freqs}\n")
    f.write(f"Model Hyperparams -> Dropout Rate: {dropoutRate}, "
            f"Kernel Length: {kernLength}, F1: {F1}, D: {D}, F2: {F2}, "
            f"Dropout Type: {dropoutType}\n")
    f.write(f"Training -> Stop threshold: {stop_threshold}, Epochs: {epochs}, Batch size: {batch_size}, Learning Rate Schedule: {lr}, L2 regulariser: {l2}\n")
    if augment:
        f.write(f"Augmentation -> {augment_chops} chops. Timeshift probs: {augment_probs[0]}, Noise addition prob: {augment_probs[1]}, Channel Dropout prob: {augment_probs[2]}\n")
    else:
        f.write("Augmentation: None\n")
    f.write("------------------------------------------------\n")

#Same-subject evaluation
if same_subject:

    # Iterate over each subject/session in the dataset
    for i in (range(subjects)):

        training_files = training_files_list[i]
        test_files = testing_files_list[i]

        # Load EEG data for the given subject depending on dataset
        if dataset=="BCI 2a":
            X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2a(training_files, test_files, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)
        elif dataset=="BCI 2b":
            X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2b(training_files, test_files, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)

        # Stratified K-Fold setup for same-subject cross-validation
        # Preserves class proportions in each fold.
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        fold_indices = []  # to store indices for each fold

        for _, test_index in skf.split(X_train_raw, Y_train_raw):
            fold_indices.append(test_index)
        
        # Manually define 4-fold train/val splits (rotating folds)
        train_val_split = [
            (np.concatenate([fold_indices[2], fold_indices[3], fold_indices[1]]), fold_indices[0]),    
            (np.concatenate([fold_indices[2], fold_indices[3], fold_indices[0]]), fold_indices[1]),
            (np.concatenate([fold_indices[0], fold_indices[3], fold_indices[1]]), fold_indices[2]),
            (np.concatenate([fold_indices[2], fold_indices[0], fold_indices[1]]), fold_indices[3])     
        ]
        
        # Per-subject tracking containers
        subject_acc_list = []
        subject_class_acc_sum = np.zeros(classes)
        cm = [] # list of all CMs to make a conglomerate CM at the end of each subject

        # Cross-validation over all folds
        for fold_step, (train_index, val_index) in enumerate(train_val_split):
            training_file = training_files_list[i]
            test_file = testing_files_list[i]

            # Prepare fold-specific datasets
            X_train, X_test, X_validate, Y_train, Y_validate, Y_test, freq_bins_centers, time_window_centers = util.prepare_data(X_train_raw,  X_test_raw, Y_train_raw, Y_test_raw, sample_rate, segment_len, sample_overlap, boundary, padding, input_format, chans, samples, kernels, n_freqs, True, train_index,val_index, fold_step)
            
            # Optional Data Augmentation 
            if augment:
                X_train, Y_train= util.augment(X_train, Y_train, augment_chops, timeshift_prob=augment_probs[0], noise_prob=augment_probs[1], chan_dropout_prob=augment_probs[2])

            # Model preparation (EEGNet/Deep/Shallow etc.)
            X_train, X_validate, X_test, model, numParams, checkpointer, callbacks= util.prepare_model(X_train, X_validate, X_test, classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType, stop_threshold, input_format, model_type, freq_bins_centers, time_window_centers, n_freqs, lr, l2)

            # Train the model and record history
            fittedModelHistory = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data=(X_validate, Y_validate),callbacks=callbacks)
            
            # Make predictions and visualise the fold
            _, acc, class_acc, fold_cm = util.predict_and_visualise(X_test, Y_test, model, fittedModelHistory, names, i, logfile, sum_accuracies, gui, fold_step)


            # Log results from this fold
            subject_acc_list.append(acc)
            subject_class_acc_sum = subject_class_acc_sum + class_acc
            cm.append(fold_cm)
        
        # Aggregate metrics across folds for this subject
        subject_acc = (sum(subject_acc_list)) / folds
        subject_class_acc = subject_class_acc_sum / folds
        cm = np.sum(cm, axis=0)

       
        sum_accuracies += subject_acc
        sum_class_acc = sum_class_acc + subject_class_acc
        all_subjects_cm.append(cm)

        # Visualization and logging
        if gui:
            util.plot_confusion_matrix(None, None, names, title=f"Aggregate for subject {i+1}", cm=cm)

        # Append results to log file
        with open(logfile, "a") as f:
            f.write(f"Subject {i+1} - Overall Accuracy: {subject_acc:.4f}.\n")

            # Log per-class accuracy
            for cls_idx, cls_name in enumerate(names):
                f.write(f"    Class Accuracy {cls_name}: {subject_class_acc[cls_idx]:.4f}\n")

#Cross-subject evaluation
else:
    #initialise lists
    X_train_subject = [None] * subjects
    y_train_subject = [None] * subjects
    X_test_subject = [None] * subjects
    y_test_subject = [None] * subjects

    # iterate over each subject - each one having a turn of being eval set
    for i in range(len(training_files_list)):
        
        # extract the train and test datasets
        training_file = [training_files_list[i]]
        test_file = [testing_files_list[i]]
        if dataset=="BCI 2a":
            X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2a(training_file, test_file, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)
        elif dataset=="BCI 2b":
            X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2b(training_file, test_file, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)

        #prepare the data for training
        X_train, X_test_subject[i], X_validate, Y_train, Y_validate, y_test_subject[i], freq_bins_centers, time_window_centers = util.prepare_data(X_train_raw,  X_test_raw, Y_train_raw, Y_test_raw, sample_rate, segment_len, sample_overlap, boundary, padding, input_format, chans, samples, kernels, n_freqs, False, None,None, None)
        
        #optional augmentation
        if augment:
            X_train, Y_train= util.augment(X_train, Y_train, augment_chops, timeshift_prob=augment_probs[0], noise_prob=augment_probs[1], chan_dropout_prob=augment_probs[2])
        
        X_train_subject[i] = np.concatenate((X_train, X_validate), axis=0)
        y_train_subject[i] = np.concatenate((Y_train, Y_validate), axis=0)

    # For each held-out test subject:
    #   - Randomly sample 5 subjects for training
    #   - Use the remaining 3 subjects for validation
    #   - Train once and evaluate on the held-out subject
    sum_accuracies = 0
    for test_subject in range(subjects):
        # Pool of all subjects except the one held out for testing
        subject_pool = list(range(subjects))
        subject_pool.remove(test_subject)
        
        # Randomly pick 5 subjects for training; the rest become validation
        training_subjects = random.sample(subject_pool, 5)
        val_subjects = list(set(subject_pool) - set(training_subjects))

        # Assemble train, val, test set by concatenating trials from chosen subjects

        X_train = np.array(X_train_subject)[training_subjects]
        X_train = np.concatenate(X_train,axis=0)
        y_train = np.array(y_train_subject)[training_subjects]
        y_train = np.concatenate(y_train,axis=0)

        X_val = np.array(X_train_subject)[val_subjects]
        X_val = np.concatenate(X_val,axis=0)
        y_val = np.array(y_train_subject)[val_subjects]
        y_val = np.concatenate(y_val,axis=0)

        X_test = np.array(X_test_subject)[test_subject]
        y_test = np.array(y_test_subject)[test_subject]
        
        #prepare the specified model
        X_train, X_validate, X_test, model, numParams, checkpointer, callbacks = util.prepare_model(X_train, X_val, X_test, classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType, stop_threshold, input_format, model_type, freq_bins_centers, time_window_centers, n_freqs, lr, l2)

        #train the  model with the training and val data
        fittedModelHistory = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data=(X_val, y_val),callbacks=callbacks)
        
        #predict and visualise with trained model
        _, acc, class_acc, _ = util.predict_and_visualise(X_test, y_test, model, fittedModelHistory, names, test_subject, logfile, sum_accuracies, gui, None)
        sum_accuracies += acc
        sum_class_acc = sum_class_acc + class_acc

#get the average statistics over the whole evaluation
avg_class_acc = sum_class_acc/subjects
avg_acc = sum_accuracies/subjects
avg_cm = np.sum(all_subjects_cm, axis=0)
print("AVERAGE ACCURACY OF ALL SUBJECTS: ", avg_acc)
print("OVERALL CLASSWISE ACCURACY: ", avg_class_acc)

#record training statistics of all evaluations in the log
with open(logfile, "a") as f:
    f.write(f"Total Average Accuracy: {avg_acc:.4f}\n")
    for cls_idx, cls_name in enumerate(names):
            f.write(f"    Average Class Accuracy {cls_name}: {avg_class_acc[cls_idx]:.4f}\n")

#plot the overall aggregate confusion matrix
util.plot_confusion_matrix(None, None, names, title=f"Aggregate Confusion Matrix of ALL subjects", cm=avg_cm)
        
