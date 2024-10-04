# Alec Malenfant

# CS 46200-001 002

# Oct 03 2024

# Assignment 2 Problem 2

<br/>

# Problem Overview

Using the iris data set of assignment 1 [https://archive.ics.uci.edu/dataset/53/iris,
split .csv into
<br/>a. 80% train and 20% test data
<br/>b. 70% train and 30% test data
<br/>c. Compare the accuracy, specificity and sensitivity of a and b and with the ROC curve mention which is a better model

# Which is the better model?

In my testing, the 80/20 split barely outperformed the 70/30 split. When the ROC curves of the two models were graphed the curves would often overlap one another. However, according to chapter 19 of Artificial Intelligence: A Modern Approach, 4th edition Published (ISBN-13: 9780134610993), we know that as a model gets more trainaing data it becomes more accurate. Therefore the 80/20 split model should outperform the 70/30 split model. And even though the lines do overlap most of the time, if the program is run repeatedly we can observe that the 80/20 split will repeatedly get a perfect score of 100% true positive rate and 0% false positive rate.

# Solution Overview

There are 2 main python files in this solution

- Assignment02-Problem02.py
- KNN.py

**Run Assignment02-Problem02.py to see the stats and ROC curves.** KNN is a package I wrote used to classify the test data using K nearest neighbor.

When the program is ran, the console will print:

- Training to testing data split ratio
- Index of k values
- Array of accuracy values for every value of k
- Array of True Positive Rate (TPR) values for every value of k
- Array of False Positive Rate (FPR) values for every value of k

For both the 80% and the 70% data split.

Assignment02-Problem02.py will also create a ROC curve for each data split. For each curve, a dot on the curve indicates the FPR vs TPR rate for a single value of k. **The program evaluates these rates for multiple values of k per data split** and then plots them using matplotlib.

## iris.csv

An 'iris.csv' file must be in the same directory as Assignment02-Problem02.py. This file should have at least 3 columns:

- 'petal_length' of type float
- 'petal_width' of type float
- 'species' of type string

## KnnClassifier class diagram

<br/>+-----------------------------------------------------+
<br/>| KNN Package |
<br/>+-----------------------------------------------------+
<br/>| KnnClassifier |
<br/>+-----------------------------------------------------+
<br/>| - test_data: np.ndarray |
<br/>| - train_data: np.ndarray |
<br/>| - K: int |
<br/>| - Accuracy: float |
<br/>| - TPR: float |
<br/>| - FPR: float |
<br/>+-----------------------------------------------------+
<br/>| + **init**(test_data, train_data, K) |
<br/>| + print_data() |
<br/>| + print_stats(self, accuracy_array, tpr_array, fpr_array, split)|
<br/>| + create_euclidean_distance_array(test_point) |
<br/>| + classify(test_stripped) |
<br/>| + calculate_tpr_fpr(predicted_values, actual_values)|
<br/>+-----------------------------------------------------+
