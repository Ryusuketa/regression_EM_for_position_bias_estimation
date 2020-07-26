import unittest
import numpy as np
from numpy.testing.utils import assert_equal
import pandas as pd
from simulate.generate_data_with_random import UserDocumentDataGenerator, ClickExposureDataGenerator


class TestUserDocumentDataGenerator(unittest.TestCase):
    def setUp(self):
        self.users = 100
        self.documents = 10
        self.document_segments = 2
        self.feature_dim = 20
        self.generator = UserDocumentDataGenerator(self.users, self.documents, self.document_segments, self.feature_dim)
        return self

    def test_generate_document_feature_model_parameters(self):
        results = self.generator._generate_document_feature_model_parameters(self.document_segments, self.feature_dim)
        self.assertEqual(type(results), dict)

    def test_generate_document_features(self):
        document_segment_maps = pd.DataFrame(dict(
            document=[0, 1, 2, 3, 4, 5], segment=[0, 0, 0, 1, 1, 1]))
        param_dicts = {0: np.array([1, 0]), 1: np.array([0, 1])}
        feature_dim = 2
        results = self.generator._generate_document_features(document_segment_maps, feature_dim, param_dicts)
        expects = np.array([[1000, 0]] * 3 + [[0, 1000]] * 3)
        np.testing.assert_equal(results, expects)

    def test_cross_join_user_document_features(self):
        document_features = np.array([[1000, 0]] * 3 + [[0, 1000]] * 3)
        user_features = np.eye(5)

        results = self.generator._cross_join_user_document_features(user_features, document_features)
        expects = np.array([[1, 0, 0, 0, 0, 1000, 0], [0, 1, 0, 0, 0, 0, 1000]])
        np.testing.assert_equal(results[[0, 1], [0, 4], :], expects)

    def test_generate_data(self):
        self.generator.generate_data()


class TestClickExposureDataGenerator(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.high_relevance = 0.8
        self.low_relevance = 0.1
        self.high_exposure = 0.8
        self.low_exposure = 0.1
        self.user_document = TestUserDocumentDataGenerator().setUp().generator
        self.positions = 20
        self.pertubate_rate = 0.5
        self.random_exposure_rate = 2.0
        self.generator = ClickExposureDataGenerator(self.high_relevance, self.low_relevance, self.high_exposure,
                                                    self.low_exposure, self.user_document, self.positions,
                                                    self.pertubate_rate, self.random_exposure_rate)

    def test_generate_probabilies(self):
        user_document_pairs = np.array([[0, 1], [0, 2], [1, 0]])
        results_rel, results_exp = self.generator._generate_probabilities(user_document_pairs)
        self.assertEqual(results_rel.shape, results_exp.shape)

    def test_generate_data(self):
        results_rel, results_exp, _, _, results_implicit_feedback = self.generator.generate_data()
        self.assertEqual(results_rel.shape, results_exp.shape)
        self.assertEqual(int(np.sum(results_implicit_feedback)), np.sum(results_implicit_feedback))
