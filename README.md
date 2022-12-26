# Penguin Classification
In `preprocessing and analysis.ipynb`, we preprocess the data and compare various classfications methods. Major classifiers compared are SVC, KNN classifier, Decision tree classifier, Random forest classifier, AdaBoost classifier. These models are compared based on their accuracies, recall rate, precision rate, F1 score and ROC curve. Also we train and compare OneVsRestClassifier and OneVsOneClassifier.


To improve the accuracy of the models, we performed *feature engineering* for discrete and continuous variables and *feature selection* using correlation matrix. 
Also since the training data had less samples, we did *data augmentation* to increase the number of samples.

`classifier.py` is a Python script which takes the file `penguin_train.csv` and trains a linear support vector classifier on it. The classifier is then used to predict the species of the penguins given a feature vector.
