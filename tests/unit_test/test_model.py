from src.model import RegressionBasedEM, _expand_exposure_probability, _expand_relevance_probability
import numpy as np
import unittest


class TestRegressionBasedEM(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        position = 3
        query = 5
        document = 10
        c = np.zeros([position, query, document])
        c[0, 0, 0], c[0, 1, 5], c[0, 2, 3], c[1, 3, 8], c[2, 4, 9] = [1] * 5
        e = np.zeros([position, query, document])
        e[np.arange(3), 0, [0, 1, 2]] = 1
        e[np.arange(3), 1, [5, 4, 3]] = 1
        e[np.arange(3), 2, [3, 1, 2]] = 1
        e[np.arange(3), 3, [7, 8, 2]] = 1
        e[np.arange(3), 4, [8, 0, 9]] = 1
        f = np.ones([query, document, 10])
        self.model = RegressionBasedEM(
            position=position, query=query, document=document, clicks=c, item_exposure=e, query_document_features=f
        )

    def test_get_features(self):
        indices = (np.array([0, 0, 0, 0]), np.arange(4), np.arange(4))
        results = self.model._get_features(indices)
        self.assertEqual(results.shape, (4, 10))

    def test_sample_relevance(self):
        indices = [0, 2, 3]
        probabilites = np.array([0.5, 0.1, 1., 0.])
        results = self.model._sample_relevance(indices, probabilites)
        expects = np.array([0, 1, 0])
        np.testing.assert_equal(results, expects)

    def test_get_all_feature_and_relevance_label(self):
        results = self.model._get_all_feature_and_relevance_label()
        self.assertEqual(results[0].shape, (140, 10))
        self.assertEqual(results[1].shape, (140,))

    def test_flatten_feature_array(self):
        self.model._query = 3
        self.model._document = 5
        array = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]])
        results = self.model._flatten_feature_array(array)
        expects = np.arange(1, 16)[:, np.newaxis]
        np.testing.assert_equal(results, expects)

    def test_reshape_relevance_array(self):
        self.model._query = 3
        self.model._document = 5
        array = np.arange(1, 16)
        results = self.model._reshape_relevance_array(array)
        expects = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        np.testing.assert_equal(results, expects)

    def test_expand_exposure_probability(self):
        size = (3, 5)
        array = np.array([1, 0, 0, 0, 0, 0])
        results = _expand_exposure_probability(array, size)
        expects = np.zeros([6, 3, 5])
        expects[0, :, :] = 1
        np.testing.assert_equal(results, expects)

    def test_expand_relevance_probability(self):
        size = (6, )
        array = np.ones([3, 5])
        results = _expand_relevance_probability(array, size)
        expects = np.ones([6, 3, 5])
        np.testing.assert_equal(results, expects)
