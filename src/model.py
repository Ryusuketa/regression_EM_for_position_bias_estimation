import numpy as np
import lightgbm as lgbm
from typing import Tuple, Dict, Any


def _expand_exposure_probability(array: np.array, size: Tuple[int]) -> np.array:
    expands_indicator = (1,) + size
    return np.tile(array[:, np.newaxis, np.newaxis], expands_indicator)


def _expand_relevance_probability(array: np.array, size: Tuple[int]) -> np.array:
    expands_indicator = size + (1, 1,)
    return np.tile(array, expands_indicator)


class RegressionBasedEM:
    def __init__(self,
                 position: int,
                 query: int,
                 document: int,
                 clicks: np.array,
                 item_exposure: np.array,
                 query_document_features: np.array,
                 gbdt_params: Dict[str, Any] = dict(max_depth=3, learning_rate=0.2, class_weight='balanced')):
        self._position = position
        self._query = query
        self._document = document
        self._probability_not_expo_rel = np.random.rand(position, query, document)
        self._probability_expo_not_rel = np.random.rand(position, query, document)
        self._probability_not_expo_not_rel = np.random.rand(position, query, document)
        self._exposure_probability = np.zeros(position)
        self._relevance_probability = np.zeros([query, document])
        self._clicks = clicks
        self._feature_vectors = query_document_features
        self._relevance_model = lgbm.LGBMClassifier(**gbdt_params)
        self._exposure = item_exposure
        self.__clicked_probability = np.ones([position, query, document])
        self._fitted = False

    def do_e_step(self):
        exposure = _expand_exposure_probability(self._exposure_probability, self._relevance_probability.shape)
        relevance = _expand_relevance_probability(self._relevance_probability, self._exposure_probability.shape)
        denominator = 1 - exposure * relevance
        not_exposure = 1 - exposure
        not_relevance = 1 - relevance

        self._probability_not_expo_rel = not_exposure * relevance / denominator
        self._probability_expo_not_rel = exposure * not_relevance / denominator
        self._probability_not_expo_not_rel = not_exposure * not_relevance / denominator

    def do_m_step(self):
        self._exposure_probability = self._calculate_exposure_probablity()
        self._relevance_probability = self._calculate_relevance()

    def _calculate_exposure_probablity(self):
        denominator = np.prod(self._relevance_probability.shape)
        clicked_components = np.sum(self._clicks, axis=(1, 2))
        not_clicked_components = np.sum(self._probability_expo_not_rel, axis=(1, 2))
        return (clicked_components + not_clicked_components) / denominator

    def _calculate_relevance(self):
        features, relevance_label = self._get_all_feature_and_relevance_label()
        self._train_gbdt(features, relevance_label)

        features = self._flatten_feature_array(self._feature_vectors)
        relevance = self.predict_relevance(features)
        return self._reshape_relevance_array(relevance)

    def _flatten_feature_array(self, features: np.array) -> np.array:
        return features.reshape(self._query * self._document, -1)

    def _reshape_relevance_array(self, relevance: np.array):
        return relevance.reshape(self._query, self._document)

    def _get_all_feature_and_relevance_label(self):
        click_indices = np.where(self._clicks == 1)
        click_features, click_relevance_label = self._get_feature_and_relevance_label(
            click_indices, self.__clicked_probability
        )
        not_exposure_indices = np.where((self._exposure == 0))
        not_exposed_features, not_exposed_relevance_label = self._get_feature_and_relevance_label(
            not_exposure_indices, self._probability_not_expo_rel
        )
        features = np.concatenate([click_features, not_exposed_features], axis=0)
        features, indices = np.unique(features, axis=0, return_index=True)
        relevance_label = np.concatenate([click_relevance_label, not_exposed_relevance_label])
        relevance_label = relevance_label[indices]
        return features, relevance_label

    def _get_features(self, indices: Tuple[np.array, ...]) -> np.array:
        return self._feature_vectors[indices[1:]]

    def _sample_relevance(self, indices: np.array, probabilities: np.array):
        target_probabilities = probabilities[indices]
        return np.random.binomial([1] * len(target_probabilities), target_probabilities, len(target_probabilities))

    def _get_feature_and_relevance_label(self, indices: Tuple[np.array, ...],
                                         probabilities: np.array) -> Tuple[np.array, np.array]:
        features = self._get_features(indices)
        labels = self._sample_relevance(indices, probabilities)
        if len(features) > 100000:
            indices = np.random.permutation(len(features))[:100000]
            features = features[indices]
            labels = labels[indices]
        return features, labels

    def _train_gbdt(self, features: np.array, labels: np.array):
        if self._fitted:
            init_score = self._relevance_model.predict(features)
            self._relevance_model.fit(features, labels, init_score=init_score)
        else:
            self._relevance_model.fit(features, labels)
            self._fitted = True

    def predict_relevance(self, features: np.array):
        gbdt_score = self._relevance_model.predict_proba(features)[:, 1]
        return gbdt_score
