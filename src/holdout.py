import numpy as np


class Holdout:
    """
    Hold-out cross-validator generator. In the hold-out, the
    data is split only once into a train set and a test set.
    Unlike in other cross-validation schemes, the hold-out
    consists of only one iteration.

    Parameters
    ----------
    n : total number of samples
    test_size : 0 < float < 1
        Fraction of samples to use as test set. Must be a
        number between 0 and 1.
    random_state : int
        Seed for the random number generator.
    """

    def __init__(self, n, test_size=0.3):
        self.n = n
        self.test_size = test_size

    def __iter__(self):
        n_test = int(np.ceil(self.test_size * self.n))
        n_train = self.n - n_test
        permutation = np.arange(self.n)
        ind_test = permutation[:n_train]
        ind_train = permutation[n_train:n_train + n_test]
        yield ind_train, ind_test

    def split(self, X=None, y=None, groups=None):
        n_test = int(np.ceil(self.test_size * self.n))
        n_train = self.n - n_test
        permutation = np.arange(self.n)

        ind_train = permutation[:n_train]
        ind_test = permutation[n_train:n_train + n_test]
        yield ind_train, ind_test

    def get_n_splits(self, *args):
        return 1
