# Fraud Detection By Machine Learning and Deep Learning

This project aims to classify the fraud and healthy transactions in the provided dataset by kaggle.

### Road Map
-Data Preprocessing and cleaning for the machine learning models.

-Machine Learning Models with SKlearn

-Deep Learning Models with TensorFlow

-SemiSupervised Models with AutoEncoder

-Unsupervised methods, Anomaly Detection.

-Compare the results

# Scaled Data or Original Data?
Let's see whether the scaled dataset works better for the machine learning models or the original one. 
![myimage](cm_org.png)          ![myimage](cm_scaled.png) 

The left confusion matrix represents the original dataset, and the right one is for the scaled dataset. As we can see, the results for the scaled dataset has fewer false negative. Then, we use the scaled dataset for the rest of the project.

## Labels distribution

The data for the targets are not balanced. 284315 for no fraud transactions and 492 for fraud. To use the supervised method, we must apply undersampling or oversampling techniques to the dataset.
