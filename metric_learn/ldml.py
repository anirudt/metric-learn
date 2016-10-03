"""
Logistic Determinant Metric Learning. (Matthieu Guillaumin, 2009)

In this paper we present two methods for learning robust dis- tance measures: (a) a logistic discriminant approach which learns the metric from a set of labelled image pairs (LDML) and (b) a nearest neighbour approach which computes the probability for two images to belong to the same class (MkNN).

This implementation is specifically for LDML.
"""
from __future__ import print_function, absolute_import


class LDML(BaseMetricLearner):
    def __init__(self, tol=1e-3, max_iter=1000, verbose=False):
        """ Initialize the LDML learner.
        
        Parameters
        ----------
        tol: float, optional
        max_iter: int, optional
        verbose: bool, optional"""
        self.params = {
                'tol': tol,
                'max_iter': max_iter,
                'verbose': verbose
                }

    def _prepare_inputs(self, X, labels):
        """ Initialize the requisite working matrices
        based on the shape of the the input training set and the labels"""

    def fit(self, X, labels):
        """Fits the learner"""

    def transform(self, X):
        """ Transforms the given matrix according to the learned model."""
        if X is None:
            X = self.X
        return self.L.dot(X.T).T

    def transformer(self):
        return self.L
