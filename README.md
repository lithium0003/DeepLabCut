# DeepLabCut for TPU
Markerless pose estimation of user-defined features with deep learning for all animals, including humans

for Google Colaboratory TPU

[Original README.md](README.org.md)

# Modification
* Change model for tf.data.Dataset and tf.estimator
* Change training dataset for TPU
  + Fix image size
  + Use tensorflow's image distortion functions
* Change training dataset for improve performance
  + Training images are randomly rotated
  + Training images are randomly changed brightness and contrast
* Training steps are visualized for tensorboard
* For TPU, 'analyze_videos' function are changed use tfrecord converted inputs
* For TPU, add 'convert_analyze_videos' function for convert to tfrecord input
* Change 'create_labeled_video' function for alpha labels marking

# How to use in Cloaboratory TPU
1. process to label_frames steps.
1. open [Colaboratory notebook](dlc_tpu.ipynb) and follow the instructions

# How to use in local system
same as original
