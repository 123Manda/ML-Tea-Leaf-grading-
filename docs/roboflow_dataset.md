# Roboflow Dataset Management

## Project
Smart Tea Leaf Grading

## Roboflow Workspace
mandara-pabasara

## Roboflow Project
smart-tea-leaf-grading

## Dataset Version
Version 1

## Classification Type
Single-Label Classification

## Classes
- Best
- Below_Best
- Poor

## Dataset Distribution
- Training: 1,159 images
- Validation: 247 images
- Test: 250 images
- Total: 1,656 images

## Roboflow Usage
Roboflow is used for dataset organization, class verification, train-validation-test split management, and dataset versioning.

The final machine-learning classification pipeline does not use a Roboflow-trained model. Images are processed using OpenCV-based segmentation and handcrafted feature extraction. The extracted color, shape, size, and texture features are supplied to the machine-learning classifiers. SVM, Random Forest, and XGBoost were evaluated, and XGBoost was selected as the final classifier.

## Version Configuration
- Preprocessing: None
- Augmentation: None

This preserves the same cleaned and leakage-checked dataset used in the local machine-learning experiments.
