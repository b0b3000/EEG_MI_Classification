import tensorflow as tf

# List all physical GPUs detected by TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

# Check if TensorFlow is using GPU for computations
if gpus:
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is using CPU.")

