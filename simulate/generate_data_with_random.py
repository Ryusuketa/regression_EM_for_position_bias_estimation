import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class UserDocumentDataGenerator:
    def __init__(self, users: int, documents: int, document_segments: int, feature_dim: int) -> None:
        self._user = users
        self._document = documents
        self._document_segments = document_segments
        self._feature_dim = feature_dim
        self.user_segment_map, self.document_segment_map =\
            self._generate_segment_pairs(users, documents, document_segments)

    @staticmethod
    def _generate_segment_pairs(users: int, documents: int, document_segments: int):
        user_df = pd.DataFrame(dict(user=np.arange(users), segment=np.random.randint(0, document_segments, users)))
        document_df = pd.DataFrame(dict(
            document=np.arange(documents), segment=np.random.randint(0, document_segments, documents)
        ))
        return user_df, document_df

    @staticmethod
    def _generate_document_feature_model_parameters(document_segments: int, feature_dim: int) -> Dict[int, np.array]:
        def _generate_multinomial_dist_param() -> np.array:
            random = np.random.exponential(1, feature_dim)
            return random / np.sum(random)
        return dict([(i, _generate_multinomial_dist_param()) for i in range(document_segments)])

    @staticmethod
    def _generate_document_features(document_segment_map: pd.DataFrame, feature_dim: int,
                                    distribution_params_dict: Dict[int, np.array]) -> np.array:
        def _generate_feature(distribution_params: np.array) -> np.array:
            return np.random.multinomial(1000, distribution_params, size=1)
        feature_list = document_segment_map['segment'].apply(lambda x: _generate_feature(distribution_params_dict.get(x))).tolist()
        document_features = np.concatenate(feature_list, axis=0)

        assert document_features.shape == (document_segment_map.shape[0], feature_dim)
        return document_features

    @staticmethod
    def _generate_user_feature_with_one_of_k(user: int) -> np.array:
        user_features = np.eye(user)

        assert user_features.shape == (user, user)
        return user_features

    @staticmethod
    def _cross_join_user_document_features(user_features: np.array, document_features: np.array) -> np.array:
        user_document, document_user = np.meshgrid(np.arange(user_features.shape[0]),
                                                   np.arange(document_features.shape[0]))
        features = np.concatenate([user_features[user_document.T, :], document_features[document_user.T, :]], axis=2)
        assert features.shape == (user_features.shape[0], document_features.shape[0],
                                  (user_features.shape[1] + document_features.shape[1]))
        return features

    def generate_data(self):
        distribution_params_dict = self._generate_document_feature_model_parameters(self._document_segments,
                                                                                    self._feature_dim)
        document_features = self._generate_document_features(self.document_segment_map, self._feature_dim,
                                                             distribution_params_dict)
        user_features = self._generate_user_feature_with_one_of_k(self._user)
        user_document_feature = self._cross_join_user_document_features(user_features, document_features)

        return user_document_feature


class ClickExposureDataGenerator:
    def __init__(self,
                 high_relevance: float,
                 low_relevance: float,
                 high_exposure: float,
                 low_exposure: float,
                 user_document: UserDocumentDataGenerator,
                 positions: int,
                 perturbate_rate: float,
                 random_exposure_rate: float,
                 exposure_bias: bool = False) -> None:
        self._high_relevance = high_relevance
        self._low_relevance = low_relevance
        self._high_exposure = high_exposure
        self._low_exposure = low_exposure
        self._user_document = user_document
        self._decay_factor = 0.1
        self._positions = positions
        self._users = user_document.user_segment_map.shape[0]
        self._documents = user_document.document_segment_map.shape[0]
        self._perturbate_rate = perturbate_rate
        self._random_exposure_rate = random_exposure_rate
        self._exposure_bias = exposure_bias

    def _get_user_relevance_document_pairs(self) -> np.array:
        user, document = self._user_document.user_segment_map, self._user_document.document_segment_map
        user_document_pairs = pd.merge(document, user, on='segment')[['user', 'document']].to_numpy()
        return user_document_pairs

    @staticmethod
    def _get_position_decay_factor(positions: int, users: int, documents: int, decay_factor: float) -> np.array:
        position_array = np.tile(np.arange(1, positions + 1).reshape(-1, 1, 1), (1, users, documents))
        return np.exp(-decay_factor * position_array)

    @classmethod
    def _generate_rand_pairs_with_random(cls, user_document_pairs: np.array, random_exposure_rate: float, users: int, documents: int) -> np.array:
        pairs_size = len(user_document_pairs)
        random_exposure_pairs = int(pairs_size * random_exposure_rate)
        random_pairs = np.array([np.random.randint(0, users, random_exposure_pairs),
                                 np.random.randint(0, documents, random_exposure_pairs)]).T
        return random_pairs

    @classmethod
    def _generate_rand_pairs_with_bias(cls, user_document_pairs: Optional[np.array], random_exposure_rate: Optional[float], users: int, documents: int) -> np.array:
        perm = np.random.permutation(documents)
        pairs_list = [cls._generate_user_document_pairs(cls._select_users_with_binomial(users, 1 / (n + 1 + 0.5)), k)
                      for n, k in enumerate(perm)]
        return np.concatenate(pairs_list, axis=1)

    @staticmethod
    def _select_users_with_binomial(users: int, probability: float) -> np.array: 
        return np.arange(users)[np.random.binomial(1, probability, users) == 1]

    @staticmethod
    def _generate_user_document_pairs(users: np.array, document: int) -> np.array:
        users = users[:, np.newaxis]
        users[:, 1] = document
        return users

    @classmethod
    def _perturbate_pairs(cls, user_document_pairs: np.array, perturbation_rate: float, random_exposure_rate: float, users: int, documents: int, exposure_bias: bool) -> np.array:
        pairs_size = len(user_document_pairs)
        remain_size = int(pairs_size * perturbation_rate)
        rand_indices = np.random.permutation(pairs_size)[:remain_size]
        user_document_pairs_dropped = user_document_pairs[rand_indices]
        generate_func = (cls._generate_rand_pairs_with_bias if exposure_bias else cls._generate_rand_pairs_with_random)
        random_pairs = generate_func(user_document_pairs, random_exposure_rate, users, documents)
        return np.concatenate([user_document_pairs_dropped, random_pairs], axis=0)

    def _generate_probabilities(self, user_document_pairs: np.array) -> Tuple[np.array, np.array]:
        def _generate_probability(pairs: np.array, high: float, low: float, use_decay_factor: bool) -> np.array:
            array = np.zeros([self._positions, self._users, self._documents])
            array[:, pairs[:, 0], pairs[:, 1]] = high
            array[array == 0] = low
            if use_decay_factor:
                return array * self._get_position_decay_factor(self._positions, self._users, self._documents,
                                                               self._decay_factor)
            else:
                return array

        relevance = _generate_probability(user_document_pairs, self._high_relevance, self._low_relevance, False)
        exposure_pairs = self._perturbate_pairs(user_document_pairs, self._perturbate_rate, self._random_exposure_rate,
                                                self._users, self._documents, self._exposure_bias)
        exposure = _generate_probability(exposure_pairs, self._high_exposure, self._low_exposure, True)

        return relevance, exposure

    def generate_data(self) -> Tuple[np.array, ...]:
        user_document_pairs = self._get_user_relevance_document_pairs()
        relevance, exposure = self._generate_probabilities(user_document_pairs)
        click = relevance * exposure
        exposure_labels = np.random.binomial(1, exposure)
        implicit_feedback = exposure_labels * np.random.binomial(1, click)
        return relevance, exposure, click, exposure_labels, implicit_feedback
