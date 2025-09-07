import util
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import datetime

import tensorflow
print(tensorflow.__version__)

title = "EEGNet 4,2 with Paper Config"

# BIG 3
input_format = "timeseries"
model_type = "EEGNet"
dataset = "BCI 2a"

if dataset == "BCI 2a":
    tmin, tmax = 0.5, 2.5 # seconds before and after stimulus we want to record  # From the paper describing dataset 2A
    classes = 4
    training_files_list = ["A01T", "A02T", "A03T","A04T", "A05T", "A06T", "A07T", "A08T", "A09T"]
    #training_files_list = ["A01T", "A02T"]
    testing_files_list = ["A01E", "A02E", "A03E", "A04E", "A05E", "A06E", "A07E", "A08E", "A09E"]
    #testing_files_list = ["A01E", "A02E"]
elif dataset == "BCI 2b":
    training_files_list = [['B0101T','B0102T','B0103T'],['B0201T','B0202T','B0203T'],['B0301T','B0302T','B0303T'],['B0401T','B0402T','B0403T'],['B0501T','B0502T','B0503T'],['B0601T','B0602T','B0603T'],['B0701T','B0702T','B0703T'],['B0801T','B0802T','B0803T'],['B0901T','B0902T','B0903T']]
    testing_files_list = [['B0104E','B0105E'],['B0204E','B0205E'],['B0304E','B0305E'],['B0404E','B0405E'],['B0504E','B0505E'],['B0604E','B0605E'],['B0704E','B0705E'],['B0804E','B0805E'],['B0904E','B0905E']]
    tmin, tmax = 0, 3.996 
    classes = 2
else:
    print("INVALID DATASET")
    
#PREPROCESSING
bandpass = [4,40] # paper & best [4,40]
baseline = None #paper and best None
amplitude_magnification = 1000 #best = 1000
ica = False # Paper = off

#Only relevant to STFT 
segment_len=128
sample_overlap=64
boundary=None
padding=True

#Only relevant to WAVELET
n_freqs = 30

#MODEL HYPERPARAMS
dropoutRate = 0.5 #0.5 suggested by paper. 0.25 suggeted suggested for cross subject. Best 0.4
kernLength = 32 #32 sugggested by paper. Best 32
F1 = 4 # 4, 8. Best 4
D = 2 # 2. Best 4
F2 = 8 # F1 * D suggested by paper. Best 16
dropoutType = 'Dropout' # Paper suggested Dropout. Best dropout
stop_threshold = 150 # 150 best
batch_size = 64 #64 best
epochs = 1000 # 1000 best
lr = False #Learning rate scheduler yes or no. Paper = no, best = no
l2 = None # none if off. Paper = None, best = 0.1

augment = False #Best = true
augment_chops = 5 # best = 5
augment_probs= [0.5,0.5,0.3] #best = 0.5,0.5,0.3

gui=True

#Same or cross subject
same_subject = True
if same_subject:
    logfile = "samesubject_cv_log.txt"
else:
    logfile = "crosssubject_log.txt"

loops = 10
for i in range(loops):

    # Non tunable definitions
    sum_accuracies = 0
    sum_class_acc = np.zeros(4)
    all_subjects_cm = []
    subjects = len(training_files_list)

    with open(logfile, "a") as f:
        f.write("\n------------------------------------------------\n")
        if same_subject:
            f.write(f"SAME SUBJECT {title}")
        else:
            f.write(f"CROSS SUBJECT {title}")
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


    if same_subject:

        for i in (range(subjects)):

            training_files = [training_files_list[i]]
            test_files = [testing_files_list[i]]

            if dataset=="BCI 2a":
                X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2a(training_files, test_files, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)
            elif dataset=="BCI 2b":
                X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2b(training_files, test_files, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)

            skf = StratifiedKFold(n_splits=4, shuffle=True)
            fold_indices = []  # to store indices for each fold

            for _, test_index in skf.split(X_train_raw, Y_train_raw):
                fold_indices.append(test_index)
            
            train_val_split = [
                (np.concatenate([fold_indices[2], fold_indices[3], fold_indices[1]]), fold_indices[0]),    
                (np.concatenate([fold_indices[2], fold_indices[3], fold_indices[0]]), fold_indices[1]),
                (np.concatenate([fold_indices[0], fold_indices[3], fold_indices[1]]), fold_indices[2]),
                (np.concatenate([fold_indices[2], fold_indices[0], fold_indices[1]]), fold_indices[3])     
            ]

            subject_acc_list = []
            subject_class_acc_sum = np.zeros(classes)
            cm = [] # list of all CMs to make a conglomerate CM at the end of each subject

            for fold_step, (train_index, val_index) in enumerate(train_val_split):

                X_train, X_test, X_validate, Y_train, Y_validate, Y_test, freq_bins_centers, time_window_centers = util.prepare_data(X_train_raw,  X_test_raw, Y_train_raw, Y_test_raw, sample_rate, segment_len, sample_overlap, boundary, padding, input_format, chans, samples, kernels, n_freqs, True, train_index,val_index, fold_step)
                
                if augment:
                    X_train, Y_train= util.augment(X_train, Y_train, augment_chops, timeshift_prob=augment_probs[0], noise_prob=augment_probs[1], chan_dropout_prob=augment_probs[2])

                X_train, X_validate, X_test, model, numParams, checkpointer, callbacks= util.prepare_model(X_train, X_validate, X_test, classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType, stop_threshold, input_format, model_type, freq_bins_centers, time_window_centers, n_freqs, lr, l2)
                
                fittedModelHistory = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data=(X_validate, Y_validate),callbacks=callbacks)
                
                _, acc, class_acc, fold_cm = util.predict_and_visualise(X_test, Y_test, model, fittedModelHistory, names, i, logfile, sum_accuracies, gui, fold_step)

                subject_acc_list.append(acc)
                subject_class_acc_sum = subject_class_acc_sum + class_acc
                cm.append(fold_cm)
            
            subject_acc = (sum(subject_acc_list)) / classes
            subject_class_acc = subject_class_acc_sum / classes
            cm = np.sum(cm, axis=0)

            sum_accuracies += subject_acc
            sum_class_acc = sum_class_acc + subject_class_acc
            all_subjects_cm.append(cm)

            if gui:
                util.plot_confusion_matrix(None, None, names, title=f"Aggregate for subject {i+1}", cm=cm)

            with open(logfile, "a") as f:
                f.write(f"Subject {i+1} - Overall Accuracy: {subject_acc:.4f}.\n")

                # Log per-class accuracy
                for cls_idx, cls_name in enumerate(names):
                    f.write(f"    Class Accuracy {cls_name}: {subject_class_acc[cls_idx]:.4f}\n")

    else:
        X_train_subject = [None] * subjects
        y_train_subject = [None] * subjects
        X_test_subject = [None] * subjects
        y_test_subject = [None] * subjects

        for i in range(len(training_files_list)):

            training_file = [training_files_list[i]]
            test_file = [testing_files_list[i]]
            
            if dataset=="BCI 2a":
                X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2a(training_file, test_file, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)
            elif dataset=="BCI 2b":
                X_train_raw, Y_train_raw, X_test_raw, Y_test_raw, chans, kernels, samples, names, sample_rate = util.get_bci_2b(training_file, test_file, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline, ica=ica, gui=gui)

            X_train, X_test_subject[i], X_validate, Y_train, Y_validate, y_test_subject[i], freq_bins_centers, time_window_centers = util.prepare_data(X_train_raw,  X_test_raw, Y_train_raw, Y_test_raw, sample_rate, segment_len, sample_overlap, boundary, padding, input_format, chans, samples, kernels, n_freqs, False, None,None, None)
            
            if augment:
                X_train, Y_train= util.augment(X_train, Y_train, augment_chops, timeshift_prob=augment_probs[0], noise_prob=augment_probs[1], chan_dropout_prob=augment_probs[2])
            
            X_train_subject[i] = np.concatenate((X_train, X_validate), axis=0)
            y_train_subject[i] = np.concatenate((Y_train, Y_validate), axis=0)

        sum_accuracies = 0
        for test_subject in range(subjects):
            subject_pool = list(range(subjects))
            subject_pool.remove(test_subject)
            
            training_subjects = random.sample(subject_pool, 5)
            val_subjects = list(set(subject_pool) - set(training_subjects))

            print("Training subjects: ", training_subjects)
            print("Val subjects", val_subjects)
            print("test subjects", test_subject)

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
            
            X_train, X_validate, X_test, model, numParams, checkpointer, callbacks = util.prepare_model(X_train, X_val, X_test, classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType, stop_threshold, input_format, model_type, freq_bins_centers, time_window_centers, n_freqs, lr, l2)

            fittedModelHistory = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data=(X_val, y_val),callbacks=callbacks)
            
            _, acc, class_acc, _ = util.predict_and_visualise(X_test, y_test, model, fittedModelHistory, names, test_subject, logfile, sum_accuracies, gui, None)
            sum_accuracies += acc
            sum_class_acc = sum_class_acc + class_acc

    avg_class_acc = sum_class_acc/subjects
    avg_acc = sum_accuracies/subjects
    avg_cm = np.sum(all_subjects_cm, axis=0)
    print("AVERAGE ACCURACY OF ALL SUBJECTS: ", avg_acc)
    print("OVERALL CLASSWISE ACCURACY: ", avg_class_acc)

    with open(logfile, "a") as f:
        f.write(f"Total Average Accuracy: {avg_acc:.4f}\n")
        for cls_idx, cls_name in enumerate(names):
                f.write(f"    Average Class Accuracy {cls_name}: {avg_class_acc[cls_idx]:.4f}\n")

    util.plot_confusion_matrix(None, None, names, title=f"Aggregate Confusion Matrix of ALL subjects", cm=avg_cm)
        
