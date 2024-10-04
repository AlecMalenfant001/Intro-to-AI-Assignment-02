"""
Alec Malenfant
This package is used in Assignment02-Problem02.py to classify the Iris dataset using a K-Nearest Neighbors (KNN) algorithm.
"""
import numpy as np
from collections import Counter

class KnnClassifier:
    def __init__(self, test_data, train_data, K=1):
        """
        Args:
            test_data: structured array with 'petal length', 'petal width', and 'species'
            train_data: structured array with 'petal length', 'petal width', and 'species'
        """ 
        self.test_data = test_data
        self.train_data = train_data
        self.K = K
        self.accurcy = 0.0
        self.TPR = 0.0  # True Positive Rate
        self.FPR = 0.0  # False Positive Rate

    def print_data(self):
        print("FROM KnnClassifier : ")
        print(self.test_data)
        print(self.train_data)

    def print_stats(self, accuracy_array, tpr_array, fpr_array, split):
        print("\n" + str(int(100*split)) + "% Train / " + str(int(100*(1-split))) + "% Test")
        print("Index : [k=1, k=3, k=5, k=7, k=9]")
        print("Accuracy : " + str(accuracy_array))
        print("TPR Array : " + str(tpr_array))
        print("FPR Array : " + str(fpr_array))

    """
    create_euclidean_distance_array(self, test_point)

    Calculates the Euclidean distances between a given test point and all points in the 'test_data' array.

    **Arguments:**
    - `self`: The instance of the class containing the 'train_data' attribute.
    - `test_point`: A tuple or list representing the coordinates of the test point.

    **Returns:**
    - A NumPy array containing the calculated Euclidean distances for each point in 'train_data', sorted in ascending 
        order. The array has columns for 'petal_length', 'petal_width', 'species' and 'distance'.
    """

    def create_euclidean_distance_array(self, test_point):
        # Copy the 'petal_length' and 'petal_width' values from the points array to a new array
        results_dtype = [('petal_length', 'f8'),('petal_width', 'f8'),('species','U20'), ('distance', 'f8')]
        results = np.empty(len(self.train_data), dtype=results_dtype)
        results['petal_length'] = self.train_data['petal_length']
        results['petal_width'] = self.train_data['petal_width']
        results['species'] = self.train_data['species']

        # Calculate the Euclidean distances
        distances = np.sqrt(
            (self.train_data['petal_length'] - test_point[0]) ** 2 + (self.train_data['petal_width'] - test_point[1]) ** 2)
        results['distance'] = distances

        # Sort the results by distance
        results = np.sort(results, order='distance')

        return results


    """
    classify(self, test_stripped)

    Classifies test data points using a K-Nearest Neighbors (KNN) algorithm.

    Parameters:
        - self: An instance of the KnnClassifier class, providing access to the training data and K value.
        - test_stripped: A structured array with columns 'petal_width', 'petal_length', and 'species' with the species
                            column blank. 

    Returns:
        - A modified structured array of test data points, with each point now containing an additional 'species' key 
            indicating the predicted class.
    """
    def classify(self, test_stripped):
        # Loop through every test data point
        for i in range(len(test_stripped)):
            test_point = (test_stripped[i]['petal_length'],test_stripped[i]['petal_width'])
            # calculate euclidean distances from test point to every point in the training set
            distances = KnnClassifier.create_euclidean_distance_array(self=self, test_point=test_point)

            # find the k closest points to test point
            closest_points = distances[:self.K]

            # find the most common class
            counter = Counter(closest_points['species']) # Create a Counter object

            # Get the most common value and its count
            most_common_species, most_common_count = counter.most_common(1)[0]

            # Set Species
            test_stripped[i]['species'] = most_common_species

        return test_stripped


    def calculate_acc_tpr_fpr(self, predicted_values, actual_values):
        # calculate true positive rate and false positive rate  using the one vs rest method
        speciesList = ['Iris-versicolor','Iris-virginica','Iris-setosa']
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for species in speciesList:
            for i in range(len(predicted_values)):

                if actual_values['species'][i] == species:
                    # Positive
                    if predicted_values['species'][i] == species:
                        true_positive += 1
                    else:
                        false_positive += 1
                else:
                    if predicted_values['species'][i] == species:
                        false_negative += 1
                    else:
                        true_negative += 1

        # Calculate Accuracy, TPR and FPR
        self.accurcy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        self.TPR = (true_positive) / (true_positive + false_negative)
        self.FPR = (false_positive) / (false_positive + true_negative)

        #return np.array([self.FPR, self.TPR])
        return self.accurcy, self.FPR, self.TPR



