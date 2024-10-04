# Alec Malenfant
# Intro to AI Assignment 02 Problem 01
# Given a list of coordinates on an XY-Plane
# And an integer k
# This program will return the k closest points to the origin
import numpy as np

""" calculate_distances_with_structured_array(points) Calculates the distance from the origin to each point in the 
        array. Returns a structured array.
    Args:
        points: A 2D NumPy array of points.
    Returns:
        A NumPy structured array where each element contains the point and its distance from the origin.
"""


def calculate_distances_with_structured_array(points):
    dtype = [('point', 'f8', 2), ('distance', 'f8')]
    distances = np.empty(len(points), dtype=dtype)
    distances['point'] = points
    distances['distance'] = np.linalg.norm(points, axis=1)  # Calculate euclidean distance from origin
    return distances

if __name__ == '__main__':
    print("-- K-Nearest-Neighbor --")
    # Ask user to define Points
    print("Please input all points at once as a 2d Array. For example : [[3,3],[5,-1],[-2,4]]")
    input_str = input(": ")

    # Parse the input string
    input_str = input_str.strip('[]')
    points_list = [list(map(float, point.split(','))) for point in input_str.split('],[')]
    points = np.array(points_list)

    # Ask user to define K
    K = int(input("\nPlease define k. Enter an integer less than or equal to " + str(len(points)) + ": "))

    # Create structured array of points and their distances to the origin
    structured_distances = calculate_distances_with_structured_array(points)

    # Sort the structured array by distance
    sorted_distances = structured_distances[np.argsort(structured_distances['distance'])]

    # Print the K closest points
    print("\nThe closest " + str(K) + " points to the origin in descending order are : ")
    print(sorted_distances['point'][:K])

    # Keep window open
    input("\nPress enter to quit ")
    exit()


