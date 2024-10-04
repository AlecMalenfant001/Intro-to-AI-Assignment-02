# Alec Malenfant

# CS 46200-001 002

# Oct 03 2024

# Assignment 2 Problem 1

<br/>

# Problem Overview

Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0). The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)^2 + (y1 - y2)^2).

## Example 1:

Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].

## Example 2:

Input: points = [[3,3],[5,-1],[-2,4]], k = 2
Output: [[3,3],[-2,4]]
Explanation: The answer [[-2,4],[3,3]] would also be accepted.
These are examples. The problem is user input based. The grader can test with any input of their choice.

## Constraints:

• 1 <= k <= points.length <= 104
• -104 <= xi, yi <= 104

# Input

## Array of points

The program will ask the user for two inputs. The first is a 2d array of points. Each point should be in the form [x,y], and the entire array should be in the form [[x1,y1],[x2,y2],...,[xn,yn]]. This value is input as a string and will be parsed by the program.

For example : [[3,3],[5,-1],[-2,4]]

## K

The second input that the program will ask from the user is an integer less than or equal to the number of points entered.
