# EEG Motor Imagery Classification Pipeline
*A Deep Learning Framework for EEG-Motor Imagery classificiation in BCI Systems*

## Overview
- End-to-end EEG motor imagery classification system built for the **BCI Competition IV 2a/2b** datasets.  
- Implements complete preprocessing, model training, evaluation, and visualization workflows for brain–computer interface (BCI) research.  
- Supports the **EEGNet**, **ShallowConvNet**, **DeepConvNet**, and newly proposed **modified EEGNet** models.
---
## Repo Structure
- The `ubuntu` branch and `main` branch contain very similar code, with `ubuntu` optimised for running on Ubuntu/Linux systems.
---

## Pipeline Summary
1. **Data Loading** - collect data from datasets specified
2. **Preprocessing** – bandpass filtering, ICA artifact removal, epoching, moving standardisation, augmentation, and more.
3. **Model Setup** – dynamically construct and compile CNN architecture with specified parameters  
4. **Training** – Fit the model on training and validation data with callbacks and checkpoints.  
5. **Evaluation** – Computes accuracy, per-class results, confusion matrices, confidence plots, and more.

---

## Evaluation Modes
- **Same-Subject Evaluation** – 4-fold evaluation on the same subject the model was trained on, with individual and averaged performance recorded.  
- **Cross-Subject Evaluation** – Train/validate on all subjects except one, and evaluate on the held-out subject. 

## Outputs
- Training logs → `/logs/crosssubject_log.txt`, `/logs/samesubject_log.txt`   
- Confusion matrices, plots and other visualisations → `/figures`  
- Other training and evaluation feedback → Command Line output

## Usage
- Configure all parameters in script.py.
- Change folder locations in the head of util.py
- Run: `python script.py`

## Key References
- [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., and Lance, B. J. 
Eegnet: a compact convolutional neural network for eeg-based brain–computer interfaces. 
Journal of neural engineering 15, 5 (2018), 056013.

- [2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, 
K., Tangermann, M., Hutter, F., Burgard, W., and Ball, T. Deep learning with convolutional neural 
networks for eeg decoding and visualization. Human brain mapping 38, 11 (2017), 5391–5420.

- [3] Labratory, A. R. arl-eegmodels. https://github.com/vlawhern/arl-eegmodels/, 2022.
