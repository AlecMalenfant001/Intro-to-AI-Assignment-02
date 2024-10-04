""" Alec Malenfant
 Intro to AI Assignment 02 Problem 02
 In this assignment, the program will split a data set into :
 80% training data & 20% test data
 and
 70% trainging data & 30% test data
 and then compare the accuracy, specificity, and sensitivity of each data split
 with an ROC curve.
"""
import numpy as np
import matplotlib.pyplot as plt
from KNN import KnnClassifier
import os

"""
SplitData(data, train_split) splits array-like objects into training and test data 
    Parameters: 
        data : array-like 
                    object to be split 
 
        train_split : float 
                    The percentage of data to become training data. All other data becomes testing data.
                    For example, testSplit=0.7 => 70% testing data and 30% training data
    Returns: two global variables
        trainData : array-like
                    A subset of input data intended to be used for model training
        testData : array-like
                    A subset of input data intended to be used for model testing

"""


def split_data(data, train_split):
    global trainData
    global testData
    split = int(len(data) * train_split)  # calculate split point
    trainData = data[:split]
    testData = data[split:]


"""
    heavy_lifting(split) 

    Description:
    This function performs a k-Nearest Neighbors (k-NN) classification on a dataset.
    It shuffles the dataset, splits it into training and test sets, 
    and calculates the Accuracy, True Positive Rate (TPR), and False Positive Rate (FPR) 
    for multiple odd values of k. The function then prints the statistics and returns them.

    Arguments:
    - split: A parameter indicating the split ratio for the dataset.

    Returns:
    - stats: A numpy array containing the accuracy, FPR, and TPR for each value of k.
    """


def heavy_lifting(split):
    # First pass : 80% train / 20% test
    k = 1
    while k <= 9:
        # Randomize the entries by shuffling the numpy array
        np.random.shuffle(data_array)

        # Split data
        split_data(data_array, .8)  # 80% training data and 20% test data

        # create subset of training_data with the 'species' column blank
        test_stripped = np.empty(len(testData), dtype=lineDType)
        test_stripped['petal_length'] = testData['petal_length']
        test_stripped['petal_width'] = testData['petal_width']

        # Predict values and calculate TPR and FPR 
        classifier = KnnClassifier(test_data=testData, train_data=trainData, K=k)
        test_stripped = classifier.classify(test_stripped)  # predict values
        accuracy, fpr, tpr = classifier.calculate_acc_tpr_fpr(predicted_values=test_stripped, actual_values=testData)

        # Add data to stats array
        if k == 1:  # Create stats array
            stats = np.array([accuracy, fpr, tpr])
        else:  # Add stats to array
            stats = np.array([np.append(stats[0], accuracy), np.append(stats[1], fpr),
                              np.append(stats[2], tpr)])  # Add stats to array

        # Increase k to the next odd number
        k += 2

    # Print Accuracy, TPRs, FPRs
    classifier.print_stats(accuracy_array=stats[0], fpr_array=stats[1], tpr_array=stats[2], split=split)

    return stats


if __name__ == '__main__':
    # Change working directory to the path of this python file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Read the CSV file into a list of rows
    with open('./iris.csv', 'r') as file:
        lines = file.readlines()

    # Remove CSV header
    lines.pop(0)

    # Split each line into a list of values and convert to a structured numpy array
    data_list = []
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        values = line.strip().split(',')
        # Convert numerical values to float and keep species as string
        data_tuple = (float(values[2]), float(values[3]), str(values[4]))
        data_list.append(data_tuple)

    # Convert the list of tuples to a structured numpy array
    lineDType = [('petal_length', 'f8'),
                 ('petal_width', 'f8'), ('species', 'U20')]
    data_array = np.array(data_list, dtype=lineDType)

    # First pass : 80% train / 20% test
    stats0 = heavy_lifting(0.8)

    # Second pass : 70% train / 30% test
    stats1 = heavy_lifting(0.7)

    # create ROC curves
    plt.figure()
    lw = 2
    plt.plot(  # Curve 1 : 80% train / 20% test
        stats0[1],
        stats0[2],
        color="darkorange",
        lw=lw,
        label="*(80% training data)",
    )
    plt.plot(  # Curve 2 : 70% train / 30% test
        stats1[1],
        stats1[2],
        color="green",
        lw=lw,
        label="*(70% training data)",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")

    # Use scatter to plot the points
    plt.scatter(stats0[1], stats0[2], color="darkorange", s=100, label="K value")
    plt.scatter(stats1[1], stats1[2], color="green", s=100, label="K value")

    plt.show()

    # Keep window open 
    input("\nPress Enter to quit ")
    plt.close()
    exit()
