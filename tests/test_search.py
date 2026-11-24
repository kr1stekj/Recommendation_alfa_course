"""Безопасное построение ссылок без интерпретации произвольных схем и путей."""
import unittest
from types import SimpleNamespace
from reco.search import material_url, SmartSearch


class MaterialUrlTests(unittest.TestCase):
    def test_platform_slug_and_existing_https(self):
        self.assertEqual(material_url({'kind': 'course', 'url': 'marketing'}), 'https://kurs.alfabank.ru/courses/marketing/')
        self.assertEqual(material_url({'url': 'https://example.org/article'}), 'https://example.org/article')
        self.assertEqual(material_url({'url': '/articles/finance/'}), 'https://kurs.alfabank.ru/articles/finance/')

    def test_unsafe_and_unknown_urls_stay_identifiers(self):
        for value in ('javascript:alert(1)', '//example.org', 'http://example.org', 'https://u:pass@example.org',
                      '/other-path', '../private', 'a/b', 'has space', 'https://[broken', '\\example.org', None):
            with self.subTest(url=value): self.assertIsNone(material_url({'kind': 'course', 'url': value}))


class OrdinarySearchVocabularyTests(unittest.TestCase):
    def search(self, descriptions):
        catalog = [dict(id='article:' + str(i), title=title, description=description,
                        topics=[], tags=[], eligible=True, kind='article',
                        url='material-' + str(i), price=None)
                   for i, (title, description) in enumerate(descriptions)]
        return SmartSearch(SimpleNamespace(catalog=catalog))

    def test_exact_word_in_only_one_material_is_found(self):
        search = self.search([('Эквайринг для бизнеса', ''), ('Налоги бизнеса', ''),
                              ('Бизнес и продажи', '')])
        result = search.search(task='эквайринг', search_mode='normal')
        self.assertEqual(result['total_matches'], 1)
        self.assertEqual(result['items'][0]['title'], 'Эквайринг для бизнеса')

    def test_rare_word_survives_large_catalog_vocabulary(self):
        # Редкий термин сортируется после более 8000 других слов: прежний лимит
        # исключал его независимо от частоты документа.
        description = ' '.join('термин' + str(i) for i in range(8100)) + ' яуникальноеслово'
        search = self.search([('Материал о бизнесе', description), ('Другой материал', '')])
        result = search.search(task='яуникальноеслово', search_mode='normal')
        self.assertEqual(result['total_matches'], 1)
        self.assertEqual(result['items'][0]['id'], 'article:0')
