"""Контракт шкал, пропусков и вкладов формулы; без БД и загрузки моделей."""
import json
import unittest

import numpy as np

from reco.ranking import normalized_popularity, score_components, score_breakdown


class RankingTests(unittest.TestCase):
    def score(self, cosine, algorithm='hybrid', **overrides):
        zero = np.zeros(len(cosine))
        parts = dict(topic=zero, format_match=zero, popularity=zero, dislike=zero, continuation=zero)
        parts.update(overrides)
        return score_components(algorithm, cosine, **parts)

    def test_hybrid_clamps_negative_cosine_without_shifting_zero_or_positive_values(self):
        factors, _, _, scores = self.score([-1, -0.5, 0, 0.4, 1, 1.000001])
        np.testing.assert_allclose(factors['cosine'], [0, 0, 0, 0.4, 1, 1])
        np.testing.assert_allclose(scores, [0, 0, 0, 0.3, 0.75, 0.75])

    def test_every_hybrid_factor_is_finite_and_bounded(self):
        values = [-1, 0, 1, 2, np.nan, np.inf, -np.inf]
        factors, parts, _, scores = self.score(values, topic=values, format_match=values,
            popularity=values, dislike=values, continuation=values)
        for name, values in factors.items():
            with self.subTest(name=name):
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(((values >= 0) & (values <= 1)).all())
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.isfinite(np.stack(list(parts.values()))).all())

    def test_popularity_and_dislike_have_requested_bounded_effect(self):
        _, contributions, weights, scores = self.score([0, 0, 0],
            popularity=[1, 0, 0], dislike=[0, 1, 0])
        np.testing.assert_allclose(scores, [0.05, -0.12, 0])
        self.assertAlmostEqual(sum(weights[name] for name in ('cosine','topic','format','popularity')), 1)
        self.assertEqual(contributions['dislike'][2], 0)

    def test_popularity_missing_invalid_and_large_counts(self):
        invalid = [{}, *[{'history_count': value} for value in (
            None, np.nan, np.inf, -np.inf, 'NaN', -1, 0, 1.5, 'bad')]]
        np.testing.assert_array_equal(normalized_popularity(invalid), np.zeros(len(invalid)))
        counts = normalized_popularity([*invalid, {'history_count': 1}, {'history_count': 9}])
        self.assertAlmostEqual(counts[-2], np.log(2) / np.log(10))
        self.assertEqual(counts[-1], 1)
        np.testing.assert_array_equal(normalized_popularity([{'history_count': 10 ** 400}]), [1])
        self.assertEqual(normalized_popularity([]).shape, (0,))

    def test_baselines_keep_raw_cosine_and_exact_topic_overlap(self):
        _, parts, _, scores = self.score([-0.8, 0.2, 1], algorithm='cosine',
            topic=[1, 0, 0], popularity=[1, 1, 1], dislike=[1, 1, 1])
        np.testing.assert_allclose(scores, [-0.8, 0.2, 1])
        self.assertEqual(set(parts), {'raw_cosine'})
        _, parts, _, scores = self.score([-0.8, 0.2, 1], algorithm='topic', topic=[1, 0.5, 0])
        np.testing.assert_allclose(scores, [1, 0.5, 0])
        self.assertEqual(set(parts), {'topic'})

    def test_breakdown_reconstructs_base_and_selection_score(self):
        _, parts, weights, scores = self.score([0.8], topic=[0.5], popularity=[1],
            format_match=[1], dislike=[1], continuation=[1])
        breakdown = score_breakdown('hybrid', parts, weights, 0, {'similarity': -0.04, 'same_format': -0.01})
        self.assertAlmostEqual(sum(breakdown['contributions'].values()), scores[0])
        self.assertAlmostEqual(breakdown['base_score'] - 0.05, breakdown['selection_score'])
        self.assertEqual(breakdown['formula'], 'hybrid-v2')
        json.dumps(breakdown, allow_nan=False)
