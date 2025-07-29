# take 25/25/50 percent of the data to train/validate/test
#X_train, X_temp, Y_train, Y_temp = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_train)
#X_validate, X_test, Y_validate, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp)

#X_test_original_reshaped = X_test_original.reshape(X_test_original.shape[0], chans, samples, kernels)
X_train = X_test_original[mask]
Y_train = Y_test[mask]


_, _, X_test, Y_test, chans, kernels, samples, names = util.get_bci_2a(
    None, test_files,
    bandpass = [4,40],
    tmin = tmin, tmax = tmax,
    mode = "gdf",
    amp_mag= amplitude_magnification #ALSO ADD BASELINE
    )

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train)

X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

#Y_train = np_utils.to_categorical(Y_train) # One hot encoding format for probabilistic classification #already been done
#Y_validate = np_utils.to_categorical(Y_validate) # One hot encoding format for probabilistic classification
Y_test = np_utils.to_categorical(Y_test) # One hot encoding format for probabilistic classification

# 2) Standardize per-channel (over time & trials)
mean  = X_train.mean(axis = (0,2), keepdims=True)
std   = X_train.std(axis = (0,2), keepdims=True)
X_train    = (X_train   - mean) / std
X_validate = (X_validate- mean) / std
X_test     = (X_test    - mean) / std

print('X_train shape:', X_train.shape)
print('y train shape', Y_train.shape)
print('X_test shape:', X_test.shape)
print('y test shape', Y_test.shape)
print('X_validation shape:', X_validate.shape)
print('y validation shape', Y_validate.shape)

###WHY IS THE Y TEST SHAPE INCORRECT 


# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = 4, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

numParams    = model.count_params()    

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1, save_best_only=True)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

class_weights = {0:1, 1:1, 2:1, 3:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer])

#########################################################################################################

# load optimal weights
model.load_weights('/tmp/checkpoint.h5')

#predict
print(model.summary())
probs = model.predict(X_test)
preds = probs.argmax(axis = -1)

acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy EEGNet: %f " % (acc))

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

# plot the confusion matrices for both classifiers

plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')


plt.figure(1)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'xDAWN + RG')

plt.show()

# Show only the first 10 samples for clarity
samples_to_plot = 10
util.plot_predicted_probs(probs, samples_to_plot)
