# CIFAR-10 Image Classification (PyTorch)

A computer vision project developed in PyTorch for classifying images from the CIFAR-10 dataset, with a strong focus on data augmentation and comprehensive model evaluation. The project is implemented in Python with the most ppular data science tools and PyTorch’s dynamic computation graph.

The main goal of this project was to implement a robust training pipeline that includes data augmentation to improve generalization and detailed performance metrics beyond simple accuracy.

---

# Features

- Multi-class image classification (10 classes)
- Custom Convolutional Neural Network (CNN) architecture
- Advanced data augmentation pipeline
- Automatic training/validation dataset splitting
- Model checkpointing (saves best model by validation loss)
- Comprehensive evaluation (Confusion Matrix, Class-wise Accuracy)
- Training metric visualization

---

# Data & Model Architecture

The project focuses on deep learning best practices for small image datasets:

### Robust Data Augmentation
To prevent overfitting and improve model robustness, the pipeline applies transformations including Random Affine (rotation/translation/scale), Random Perspective distortions, and Color Jitter (brightness/contrast/saturation) during training.

### Validation Strategy
Instead of relying solely on training loss, the system splits the training data (90/10 split) to create a dedicated validation set. This ensures that the model performance is monitored on unseen data during the training phase. In addition a separate test dataset is created with no data augmentation, for accurate testing.

### Best Model Checkpointing
The training loop actively monitors validation loss. It automatically saves the model state only when validation performance improves, ensuring the final artifact is the most generalized version, not just the result of the last epoch.

---

# Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- TensorBoard

---

# Core Systems

- Convolutional feature extraction
- Stochastic Gradient Descent (SGD) optimization
- Cross-Entropy loss calculation
- Confusion Matrix generation and plotting
- Training and Validation loss tracking
- Automated data downloading and preprocessing

---

# Build & Run

1. Clone the repository
2. Install dependencies (PyTorch, Matplotlib, Scikit-learn)
3. Run the script:
   ```bash
   python image_classification_25_26.py
