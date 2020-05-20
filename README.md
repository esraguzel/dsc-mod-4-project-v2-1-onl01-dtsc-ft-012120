## Module 4 Final Project
### Goals and Overview

In this project, the [Chest X-Ray Images (Pneumonia)] dataset on Kaggle is chosen to work on. The aim of this project is to predict whether the X-Ray images are belong to a healthy person or a pneumonia patient by applying neural network models.

The aim of to this project is to increase recall score for pneumonia images (Sensitivity) above 95% and target recall score for normal images (Specificity) above 90%.

### Data

The data obtained from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) has 5860 training images, divided into 3 fold of train, validation and test. The data is also highly imbalanced that number of pneumonia images exceeds the number of normal images nearly 3 times.

After manual redistribution of the dataset into 3 folders, train set contains 70%, test and validation and test sets contains 15% of the data and balanced shares of normal and pneumonia images.

It is important to note that initial accuracy levels of the deep learning models visibly increased after balanced re-distibution of the images into train, validation and test folders. 

### Methodology

To observe accuracy and recall scores throughout the models, 7 models applied. 

- Basic neural network model with 2 layers
- Regulatized basic neural networks model with dropout
- Convolutional neural networks model
- Deep convolutional neural networks model
- Xception
- VGG16
- VGG19 

Recall, accuracy and f1 scores are used for evaluation metrics. As the data is highly imbalanced to increase performance of the last model data augmentation is also applied to the dataset.

As the models got complicated, it is observed accuracy, sensitivity and specificity scores increased throughout the models. Also, it is noticed that data augmentation lead to rise in both recall scores for each labels and increased model performance.

### Findings

tablo 

best model conf matrix and accuracy report layers



### Recommendations
- The model should be used as a tool by medical experts and specialists which will support their diagnosis and treatment method.
- To reach higher levels of accuracy and recall score oversampling techniques should be used.



### Future Work 
- The Dataset can be enriched, and the target variables can be balanced by oversampling methods. 
- Different pretrained models can be applied to observe accuracy , f1 and recall scores. 
- The model can be trained over to detect the cause of pneumonia (bacteria or virus).

