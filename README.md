# Image Classification Project

This project implements an image classification model using PyTorch to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Project Structure

```
convnet.pth
copy_of_hw1_image_colorization_24_25.py
copy_of_hw2_simplified_object_detection_24_25.py
fftnet.pth
homework1_flowchart/
homework2_flowchart/
image_classification_25_26.py
model_cnn.pth
model_fcn.pth
background/
data/
    cifar-10-batches-py/
        batches.meta
        data_batch_1
        data_batch_2
        data_batch_3
        data_batch_4
        data_batch_5
        readme.html
        test_batch
objects/
practice/
runs/
    CIFAR10/
        events.out.tfevents.1771340162.DESKTOP-NVQ7O0J.14708.0
        events.out.tfevents.1771340780.DESKTOP-NVQ7O0J.60776.0
```

## Output

```
Input size (MB): 0.01
Forward/backward pass size (MB): 1.54
Params size (MB): 2.62
Estimated Total Size (MB): 4.17
----------------------------------------------------------------
Start Training  cuda:0
Epoch 0000: train loss 2.2255, valid loss 2.0979
Epoch 0001: train loss 2.0464, valid loss 1.9268
Epoch 0002: train loss 1.9290, valid loss 1.8416
Epoch 0003: train loss 1.8752, valid loss 1.8188
Epoch 0004: train loss 1.8312, valid loss 1.7452
Epoch 0005: train loss 1.7936, valid loss 1.7017
Epoch 0006: train loss 1.7667, valid loss 1.7047
Epoch 0007: train loss 1.7421, valid loss 1.6544
Epoch 0008: train loss 1.7178, valid loss 1.6273
Epoch 0009: train loss 1.6935, valid loss 1.6384
Finished Training
Accuracy of the network on the 10000 test images: 35 %
Accuracy for class plane is: 19.5 %
Accuracy for class car   is: 55.1 %
Accuracy for class bird  is: 25.3 %
Accuracy for class cat   is: 13.7 %
Accuracy for class deer  is: 29.8 %
Accuracy for class dog   is: 43.7 %
Accuracy for class frog  is: 49.3 %
Accuracy for class horse is: 45.7 %
Accuracy for class ship  is: 53.3 %
Accuracy for class truck is: 30.6 %
```

## Observations

The model achieved an overall accuracy of **35%** on the CIFAR-10 test dataset. The accuracy for individual classes varies significantly, with the highest accuracy for the "car" class (55.1%) and the lowest for the "cat" class (13.7%).

## Possible Improvements

To improve the model's performance, consider the following:

1. **Model Architecture**:
   - Use a deeper and more complex architecture, such as ResNet, VGG, or EfficientNet.
   - Add more convolutional layers, batch normalization, and dropout to the current architecture.

2. **Hyperparameter Tuning**:
   - Experiment with different learning rates, optimizers (e.g., Adam, RMSprop), and batch sizes.

3. **Data Augmentation**:
   - Apply techniques like random cropping, flipping, rotation, color jittering, and normalization to increase data diversity.

4. **Learning Rate Scheduler**:
   - Use a learning rate scheduler to reduce the learning rate as training progresses.

5. **Training Duration**:
   - Train the model for more epochs (e.g., 50 or 100) while monitoring the training and validation loss to avoid overfitting.

6. **Regularization**:
   - Add dropout layers and use weight decay (L2 regularization) in the optimizer.

7. **Pretrained Models**:
   - Fine-tune a pretrained model (e.g., ResNet, VGG) on the CIFAR-10 dataset.

8. **Data Preprocessing**:
   - Ensure input images are normalized using the mean and standard deviation of the dataset.

9. **Class Imbalance**:
   - Handle class imbalance using oversampling, undersampling, or class weighting.

10. **Evaluation Metrics**:
    - Use additional metrics like precision, recall, and F1-score to better understand the model's performance.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python image_classification_25_26.py
   ```

4. View the training logs in the `runs/` directory.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib

## License

This project is licensed under the MIT License.
