import unittest
import csv

"""
A simple behavior test by comparing the accuracies saved in tests.csv.
This is test will be executed when merging to the master branch (see .github/workflows/test.yml
"""


class BehaviorTest(unittest.TestCase):

    def test_perturbations(self):
        """
        Test that perturbations has not significant effect on accuracy
        """
        with open('report/tests.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            accuracies = [row for row in reader]
            accuracies = accuracies[1]  # accuracy values dropping header
            original = accuracies[1][0]
            for i in range(1, len(accuracies)):
                assert abs(float(original) - float(accuracies[i])) < 0.3


if __name__ == '__main__':
    unittest.main()
