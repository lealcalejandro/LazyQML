#!/usr/bin/env python

import unittest
from sklearn.datasets        import make_classification

def import_data():
    X, y = make_classification(
        n_samples    = 1000,
        n_features   = 30,
        n_informative= 20,
        n_redundant  = 5,
        n_repeated   = 0,
        n_classes    = 2,
        class_sep    = 2.0, # problema sencillo
        flip_y       = 0.0, # sin ruido, sencillo
        random_state = 0,
    )

    return X, y

class TestFastQSVM(unittest.TestCase):
    """Tests for `lazyqml` package."""

    def test_basic_exec(self):
        from lazyqml import QuantumClassifier
        from lazyqml.Global import Embedding, Ansatzs, Model

        X, y = import_data()

        nqubits = {4, 6, 8, 20}
        embeddings = {Embedding.DENSE_ANGLE}
        models = {Model.QSVM}
        verbose = True
        sequential = False

        # No importan
        ansatzs = {Ansatzs.TWO_LOCAL}
        layers = 2
        epochs = 10
        splits = 8
        repeats = 2
        threshold = 20
        batch_size = 8
        # cores = 1

        qc = QuantumClassifier(
            nqubits=nqubits,
            embeddings=embeddings,
            ansatzs=ansatzs,
            classifiers=models,
            numLayers=layers,
            verbose=verbose,
            sequential=sequential,
            epochs=epochs,
            # threshold=threshold,
            batchSize=batch_size)
        
        # if cores > 1: qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        # else: qc.fit(X, y)
        # qc.repeated_cross_validation(X,y,n_splits=splits,n_repeats=repeats)
        qc.fit(X, y)

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('unittest.main()')
    unittest.main()