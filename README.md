# GC-Recurrence

# Image Classification Model Training, Testing, and Feature Extraction

## Overview
This project mainly covers three core functionalities:
- Training an image classification model with 10-fold cross-validation.
- Evaluating the model on validation and test sets.
- Extracting features from whole slide images (WSIs) and generating heatmaps.

## Directory Structure
```plaintext
train.py
test2.py
feature_extractor.py
```

## Scripts

### 1. Training Script (`train.py`)
#### Functionality
- Trains an image classification model using 10-fold cross-validation.
- Evaluates performance on the final validation set.

#### Parameters
- `output_dir`: Directory to save output results.
- `num_classes`: Number of classes.
- `batch_size`: Batch size for training and validation.
- `num_epochs`: Number of epochs for training.
- `lr`: Learning rate.

#### Usage
```bash
python train.py
```

---

### 2. Testing Script (`test.py`)
#### Functionality
- Loads a trained model and evaluates it on the test dataset.
- Saves the test evaluation results.

#### Parameters
- `output_dir`: Directory to save output results.
- `model_path`: Path to the saved trained model.
- `data_path`: Path to the test dataset.
- `num_classes`: Number of classes.
- `batch_size`: Batch size for testing.

#### Usage
```bash
python test2.py
```

---

### 3. Feature Extraction Script (`feature_extractor.py`)
#### Functionality
- Extracts features from WSIs.
- Generates heatmaps for visualization.

#### Parameters
- `WSI_FOLDER`: Path to the folder containing WSIs.
- `weights_path`: Path to the pretrained model weights.
- `PATCH_SIZE`: Size of image patches to be extracted.
- `LEVEL`: Magnification level to extract patches from.
- `STRIDE`: Sliding window stride for patch extraction.
- `MODEL_TYPE`: Type of model used for feature extraction.
- `feature_num`: Number of features to extract per patch.

#### Usage
```bash
python feature_extractor.py
```

---

## Output Files

- **Training Results**:  
  Saved as `unipath_training_results_cv.csv`, containing training metrics https://github.com/qsyys/GC-Recurrence/blob/main/README.mdlike loss, accuracy, and AUC across cross-validation folds.

- **Testing Results**:  
  Saved as `unipath_test_results.csv`, including metrics such as accuracy, AUC, sensitivity, and specificity on the test set.

- **Feature Extraction Results**:  
  - Summary features for each WSI saved as `wsi_feature_summary.csv`.
  - Heatmap images saved as `{slide_name}_heatmap.jpg`.

---
## Acknowledgements
 - We would like to express our sincere gratitude to all contributors and collaborators who supported this project. Special thanks to our research  - team members for their invaluable discussions and technical insights, which greatly enhanced the development and implementation of this work.

 - We also acknowledge the open-source community and developers of libraries such as PyTorch, OpenSlide, and scikit-learn, whose efforts made it possible to build and refine our models efficiently. Finally, we are grateful to the institutions and organizations that provided the datasets and computational resources necessary for completing this project.
