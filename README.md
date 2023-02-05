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
![myimage](Results/cm_org.png)          ![myimage](Results/cm_scaled.png) 

The left confusion matrix represents the original dataset, and the right one is for the scaled dataset. As we can see, the results for the scaled dataset has fewer false negative. Then, we use the scaled dataset for the rest of the project.

## Labels distribution

The data for the targets are not balanced. 284315 for no fraud transactions and 492 for fraud. To use the supervised method, we must apply undersampling or oversampling techniques to the dataset.

# Machine Learning Results on Undersampled dataset:
After applying the undersampling method to the dataset and implementing several machine learning algorithms, we found these results on the original test dataset.

![myimage](Results/output.png) 

	                Accuracy Precision
  
Logistic Regression ===>	    80%,	  94%

Support Vector Machines	===>  84%,	  93%

Decision Trees	===>          71%,	  94%

Random Forest	===>            83%,	  96%

Naive Bayes	===>              87%,	  87%

K-Nearest Neighbor	===>      90%,	   91%


### The best model for the undersampled dataset is Random Forest. Let's Visualise the confusion matrix for the model.

![myimage](Results/random_forest.png) 

# Machine Learning Results on Oversampled dataset:
After applying the Oversampling method to the dataset and implementing several machine learning algorithms, we found these results on the original test dataset.

![myimage](Results/over.png) 

	                Accuracy Precision
  
Logistic Regression ===>	    97%,	  90%

Support Vector Machines	===>  98%,	  90%

Decision Trees	===>          99%,	  79%

Random Forest	===>            99%,	  82%

Naive Bayes	===>              97%,	  88%

K-Nearest Neighbor	===>      99%,	   81%


### The best model for the Oversampled dataset is Support Vector Machines. Let's Visualise the confusion matrix for the model.

![myimage](Results/svm.png) 

# Deep Learning Model, Deep Neural Network(DNN)

In the context of Machine Learning, it has been determined that the best performance on the original dataset can be achieved using a scaled and oversampled dataset. The aim of this report is to implement deep learning models using this optimized dataset.

### The following method will be used to achieve this goal:

1- Apply a Learning Rate Scheduler callback in TensorFlow to find the optimal learning rate for the model.

2- Implement early stopping and model checkpoint callbacks to prevent overfitting and save the model with the lowest loss respectively.

3- Train the model on both the train and validation data.

4- Evaluate the model on the original dataset and present the results by plotting a confusion matrix.

By following this method, we hope to achieve a deep learning model with improved performance on the original dataset.

![myimage](Results/dnn_oversample.png)

# SemiSupervised Methods: **AutoEncoder**

### Method: 
In the following steps, the AutoEncoder model is fit to an oversampled and scaled dataset. The model is then used to make predictions on the scaled dataset, resulting in a new set of predicted data. Finally, the most accurate Deep Learning model is utilized to identify fraud transactions on the original test data.

![myimage](Results/autoencode.png)
