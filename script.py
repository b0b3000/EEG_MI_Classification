# %%
import util

# BIG 3
input_format = "timeseries"
model_type = "EEGNet"
dataset = "BCI 2a"

if dataset == "BCI 2a":
    tmin, tmax = 0.5, 2.5 # seconds before and after stimulus we want to record  # From the paper describing dataset 2A
    classes = 4
    training_files_list = ["A01T", "A02T", "A03T","A04T", "A05T", "A06T", "A07T", "A08T", "A09T"]
    testing_files_list = ["A01E", "A02E", "A03E", "A04E", "A05E", "A06E", "A07E", "A08E", "A09E"]
elif dataset == "BCI 2b":
    training_files_list = [['B0101T','B0102T','B0103T'],['B0201T','B0202T','B0203T'],['B0301T','B0302T','B0303T'],['B0401T','B0402T','B0403T'],['B0501T','B0502T','B0503T'],['B0601T','B0602T','B0603T'],['B0701T','B0702T','B0703T'],['B0801T','B0802T','B0803T'],['B0901T','B0902T','B0903T']]
    testing_files_list = [['B0104E','B0105E'],['B0204E','B0205E'],['B0304E','B0305E'],['B0404E','B0405E'],['B0504E','B0505E'],['B0604E','B0605E'],['B0704E','B0705E'],['B0804E','B0805E'],['B0904E','B0905E']]
    tmin, tmax = 0, 3.996 
    classes = 2
else:
    print("INVALID DATASET")
    
#PREPROCESSING
bandpass = [4,40]
baseline = None
amplitude_magnification = 1000 #current best = 1000

#Only relevant to STFT 
segment_len=128
sample_overlap=64
boundary=None
padding=True
# INVESTIGATE CHANGING BOUNDARY AND PADDING -> Try to get less than 17 windows -> Maybe 14

#Only relevant to WAVELET
n_freqs = 30

#MODEL HYPERPARAMS
dropoutRate = 0.25 #0.5 suggested by paper
kernLength = 16 #32 sugggested by paper
F1 = 8 # 4 suggesed by paper
D = 4 # 2 suggested by paper
F2 = 32 # 16 suggested by paper
dropoutType = 'Dropout' # Paper suggested SpatialDropout2D
stop_threshold = 150 
batch_size = 64 
epochs = 1500


with open("accuracy_log.txt", "a") as f:
    f.write(f"------------------------------------------------\n")
    f.write(f'New Run. Dataset: {dataset}. input: {input_format}, model: {model_type}, tmin, tmax: {tmin}, {tmax} bandpass: {bandpass}, Baseline: {baseline}, Amplitude Magnification: {amplitude_magnification}, STFT -> Segment Length: {segment_len}, Sample Overlap: {sample_overlap}, Boundary: {boundary}, Padding: {padding}, Wavelet -> n_freqs: {n_freqs}, Model Hyperparams -> Dropout Rate: {dropoutRate}, Kernel Length: {kernLength}, F1: {F1}, D: {D}, F2: {F2}, Dropout Type: {dropoutType}, Stop threshold: {stop_threshold}, epochs: {epochs}, batch_size : {batch_size}\n')
    f.write(f"------------------------------------------------\n")

sum_accuracies = 0
for i in range(len(training_files_list)):

    training_files = [training_files_list[i]]
    test_files = [testing_files_list[i]]
    
    X_train, Y_train, X_test, Y_test, chans, kernels, samples, names, sample_rate = util.get_bci_2a(training_files, test_files, bandpass = bandpass,tmin = tmin, tmax = tmax,mode = "gdf",amp_mag= amplitude_magnification, baseline=baseline)
    
    X_train, X_test, X_validate, Y_train, Y_validate, Y_test, freq_bins_centers, time_window_centers = util.prepare_data(X_train, X_test,Y_train, Y_test, sample_rate, segment_len, sample_overlap, boundary, padding, input_format, chans, samples, kernels, n_freqs)
    
    X_train, X_validate, X_test, model, numParams, checkpointer, callbacks, class_weights = util.prepare_model(X_train, X_validate, X_test, classes, chans, samples, dropoutRate, kernLength, F1, D, F2, dropoutType, stop_threshold, input_format, model_type, freq_bins_centers, time_window_centers, n_freqs)
    
    fittedModelHistory = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data=(X_validate, Y_validate),callbacks=callbacks)
    
    sum_accuracies = util.predict_and_visualise(X_test, Y_test, model, fittedModelHistory, names, i, sum_accuracies)

avg_acc = sum_accuracies/9
print("AVERAGE ACCURACY OF ALL SUBJECTS: ", avg_acc)

with open("accuracy_log.txt", "a") as f:
    f.write(f"Total Average Accuracy: {avg_acc:.4f}\n")


