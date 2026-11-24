"""Регрессии импорта пропусков и безопасных косинусных расчётов."""
from decimal import Decimal
import unittest
import numpy as np

from reco.data import clean_text, make_catalog, historical_interactions, PUBLISHED
from reco.embeddings import Encoder, normalize


class MissingValuesTests(unittest.TestCase):
    def test_text_missing_is_empty_and_zero_is_preserved(self):
        for value in (None, float('nan'), np.float32('nan'), Decimal('NaN'),
                      ' NaN ', 'NAN', float('inf'), float('-inf')):
            with self.subTest(value=repr(value)):
                self.assertEqual(clean_text(value), '')
        self.assertEqual(clean_text(0), '0')
        self.assertEqual(clean_text('банан и nanotechnology'), 'банан и nanotechnology')

    def test_import_distinguishes_null_from_invalid_costs(self):
        base = dict(Id='course', Title='Курс', StatusId=PUBLISHED, Url='/course')
        for cost, expected in [(None, 0), ('0', 0), ('3990', 3990), ('nan', None),
                               (float('nan'), None), ('Infinity', None),
                               ('oops', None), ('3.5', None), (-1, None)]:
            with self.subTest(cost=repr(cost)):
                catalog, audit = make_catalog({'Courses': [dict(base, Cost=cost)]}, '2026-11-14')
                self.assertEqual(catalog[0]['price'], expected)
                self.assertEqual(audit['invalid_costs'], int(cost is not None and expected is None))

    def test_nan_metadata_does_not_crash_or_become_content(self):
        tables = {'Articles': [dict(Id='a', Title='Статья', StatusId=PUBLISHED, Url='/a',
                                   Description=float('nan'), Published=float('nan'), IsPublic='t')],
                  'ArticleEmbedItems': [dict(ArticleId='a', Order='nan', HtmlContent='nan')],
                  'Topics': [dict(Id='topic', Title='nan')]}
        catalog, audit = make_catalog(tables, '2026-11-14')
        self.assertEqual(catalog[0]['description'], '')
        self.assertIsNone(catalog[0]['published'])
        self.assertNotIn('nan', catalog[0]['text'])
        self.assertEqual(audit['dictionaries']['Topics'], [])
        self.assertTrue(np.isnan(tables['Articles'][0]['Description']))

    def test_missing_required_fields_are_rejected(self):
        base = dict(Id='a', Title='Курс', StatusId=PUBLISHED, Url='/a')
        for field in ('Id', 'Title', 'Url'):
            with self.subTest(field=field):
                catalog, _ = make_catalog({'Courses': [dict(base, **{field: float('nan')})]}, '2026-11-14')
                self.assertEqual(catalog, [])

    def test_missing_history_ids_are_not_joined_as_an_anonymous_user(self):
        tables = {'CourseUserBindings': [dict(Id='x', CourseId='a', UserId=float('nan')),
                                         dict(Id='y', CourseId=float('nan'), UserId='u')]}
        self.assertEqual(historical_interactions(tables, [{'id': 'course:a'}]), [])

    def test_encoder_accepts_missing_text_without_nan_tokens(self):
        e = Encoder({'backend': 'tfidf'})
        e.fit(['налоги бизнес', 'налоги курс', float('nan'), 'NaN', None])
        self.assertNotIn('nan', e.config['vocabulary'])
        result = e.encode([None, float('nan'), 'NaN'])
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(float(result.sum()), 0)

    def test_nonfinite_features_do_not_poison_cosine(self):
        result = normalize([[3, 4, np.nan], [np.inf, -np.inf, np.nan], [3e38, 3e38, 0]])
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[0], [.6, .8, 0])
        np.testing.assert_array_equal(result[1], [0, 0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(result[2])), 1, places=6)
        self.assertTrue(np.isfinite(result @ result.T).all())

    def test_semantic_encoder_accepts_empty_topic_dictionary(self):
        from types import SimpleNamespace
        encoder = Encoder({'backend': 'semantic'})
        encoder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=312))
        self.assertEqual(encoder.encode([]).shape, (0, 312))

    def test_missing_foreign_keys_never_join_each_other(self):
        base = dict(Id='course', Title='Курс', StatusId=PUBLISHED, Url='/course')
        tables = {'Courses': [base],
                  'ContentSettings': [dict(Id='settings', EntityId='course',
                                          EntityTypeId='8b440b1a-00c3-423a-9bae-3376475eb203', StatusId=PUBLISHED)],
                  'Topics': [dict(Id=float('nan'), Title='Чужая тема')],
                  'ContentSettingsTopics': [dict(ContentSettingsId='settings', TopicId=float('nan'))]}
        catalog, audit = make_catalog(tables, '2026-11-14')
        self.assertEqual(catalog[0]['topics'], [])
        self.assertEqual(audit['dictionaries']['Topics'], [])
        tables['Topics'] = [dict(Id='topic', Title='Чужая тема')]
        tables['ContentSettings'][0]['Id'] = float('nan')
        tables['ContentSettingsTopics'] = [dict(ContentSettingsId=float('nan'), TopicId='topic')]
        self.assertEqual(make_catalog(tables, '2026-11-14')[0][0]['topics'], [])

    def test_missing_article_subject_is_not_a_join_key(self):
        tables = {'Articles': [dict(Id='article', Title='Статья', StatusId=PUBLISHED,
                                    Url='/article', SubjectId=float('nan'))],
                  'ArticleSubjects': [dict(Id=float('nan'), Title='Чужая рубрика')]}
        self.assertEqual(make_catalog(tables, '2026-11-14')[0][0]['topics'], [])

    def test_missing_enrollment_key_does_not_attach_completed_lessons(self):
        tables = {'CourseUserBindings': [dict(Id=float('nan'), CourseId='a', UserId='u')],
                  'CourseUserLessonBindings': [dict(CourseUserId=float('nan'), Completed='2026-11-14')]}
        self.assertEqual(historical_interactions(tables, [{'id': 'course:a'}]), [])
