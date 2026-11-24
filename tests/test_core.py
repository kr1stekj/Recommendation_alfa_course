import io
import json
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reco.data import parse_copy, clean_text, canonical_label, make_catalog, historical_interactions, PUBLISHED, PUBLIC_TYPE
from reco.embeddings import Encoder
from reco.state import ProfileStore
from reco.llm import verify
from tests.pg_helpers import PostgresTestCase


class DataTests(unittest.TestCase):
    def test_copy_escapes_null_literal_and_unicode_separators(self):
        text = 'COPY public."T" ("A", "B", "C") FROM stdin;\n' + 'x\\ty\t\\N\t\\\\N\n' + 'text\u2028part\t\\nline\tok\n' + '\\.\n'
        rows = parse_copy(io.StringIO(text))["T"]
        self.assertEqual(rows[0], {"A": "x\ty", "B": None, "C": r"\N"})
        self.assertEqual(rows[1]["A"], 'text\u2028part')

    def test_html_scripts_whitespace_and_mixed_script(self):
        self.assertEqual(clean_text('<p>Привет&nbsp;мир</p><script>secret()</script>'), 'Привет мир')
        self.assertEqual(canonical_label(' Маркeтплейсы\t'), canonical_label('Маркетплейсы'))
        self.assertEqual(canonical_label('IT CRM'), 'it crm')
        self.assertEqual(clean_text('Маркeтплейсы'), 'Маркетплейсы')

    def test_catalog_excludes_drafts_deleted_closed_and_future(self):
        base = {"Id": "1", "Title": "Бизнес", "StatusId": PUBLISHED, "Url": "/x", "IsDeleted": "f"}
        rows = [base, dict(base, Id="2", StatusId="draft"), dict(base, Id="3", IsDeleted="t"), dict(base, Id="4", IsClosed="t"), dict(base, Id="5", Published="2027-01-01")]
        catalog, audit = make_catalog({"Courses": rows}, "2026-11-12")
        self.assertEqual([r["id"] for r in catalog], ["course:1"])
        self.assertEqual(sum(audit["rejected"].values()), 4)

    def test_null_price_rule_does_not_spread_to_podcasts(self):
        base = {"Id": "1", "Title": "Бизнес", "StatusId": PUBLISHED, "Url": "/x"}
        catalog, _ = make_catalog({"Courses": [base], "Podcasts": [base]}, "2026-11-12")
        self.assertEqual({r["kind"]: r["price"] for r in catalog}, {"course": 0, "podcast": None})

    def test_paid_lesson_text_never_enters_catalog(self):
        tables = {"Courses": [{"Id": "1", "Title": "Курс", "StatusId": PUBLISHED, "Cost": "99", "Description": "Аннотация", "Url": "/course"}], "LessonEmbedItems": [{"HtmlContent": "SECRET"}]}
        catalog, _ = make_catalog(tables, "2026-11-12")
        self.assertNotIn("SECRET", json.dumps(catalog))

    def test_tfidf_zero_query_is_finite(self):
        import numpy as np
        e = Encoder({"backend": "tfidf"})
        x = e.fit(["налоги бизнес", "налоги курс"])
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(float(e.encode(["zzzz"]).sum()), 0)

    def test_article_access_uses_type_not_editorial_flag(self):
        base = dict(Title="Статья", StatusId=PUBLISHED, Url="article", Content="BODY_SECRET", Description="Аннотация")
        rows = [dict(base, Id="public", TypeId=PUBLIC_TYPE, IsPublic="f"),
                dict(base, Id="paid", TypeId="9cda1148-8db6-4d29-98fc-6d67ea538fff", IsPublic="t"),
                dict(base, Id="closed", TypeId=PUBLIC_TYPE, IsPublic="t", IsClosed="t"),
                dict(base, Id="unknown", IsPublic="t")]
        catalog, _ = make_catalog({"Articles": rows}, "2026-11-15", include_closed_history=True)
        cards = {r['entity_id']: r for r in catalog}
        self.assertIn("BODY_SECRET", cards['public']['text'])
        for name in ('paid', 'closed', 'unknown'):
            self.assertNotIn("BODY_SECRET", cards[name]['text'])
        self.assertFalse(cards['closed']['eligible'])
        self.assertFalse(cards['unknown']['llm_public_metadata'])

    def test_access_trial_revoked_and_missing_are_not_owned(self):
        rows = [dict(Id=str(i), CourseId='c', UserId='u', ContentAccess=access)
                for i, access in enumerate(('0', '1', '2', '10', None))]
        history = historical_interactions({'CourseUserBindings': rows}, [{'id': 'course:c'}])
        self.assertEqual([r['owned'] for r in history], [False, True, True, False, False])


class ProfileTests(PostgresTestCase):
    def setUp(self):
        super().setUp()
        self.store = ProfileStore(self.database, self.dataset, max_guests=2, idle_ttl=10, absolute_ttl=25)

    @staticmethod
    def populated(state):
        state["items"]["course:1"] = {"weight": 3, "at": 100}
        return state

    def test_expired_read_and_cleanup(self):
        self.store.mutate("anon:a", "anonymous", self.populated, now=100)
        self.assertTrue(self.store.read("anon:a", now=109)["items"])
        self.assertFalse(self.store.read("anon:a", now=110)["items"])
        self.assertEqual(self.store.cleanup(now=110), 1)

    def test_active_guest_still_has_absolute_expiry(self):
        for now in (100, 109, 118, 124):
            self.store.mutate("anon:a", "anonymous", self.populated, now=now)
        self.assertFalse(self.store.read("anon:a", now=125)["items"])

    def test_global_guest_cap_and_registered_persistence(self):
        self.store.mutate("user:u", "registered", self.populated, now=100)
        for i in range(3):
            self.store.mutate("anon:" + str(i), "anonymous", self.populated, now=101 + i)
        self.assertFalse(self.store.read("anon:0", now=104)["items"])
        self.assertTrue(self.store.read("user:u", now=10000)["items"])

    def test_index_rebuild_invalidates_old_profiles(self):
        self.store.mutate("anon:a", "anonymous", self.populated, now=100)
        other = ProfileStore(self.database, self.dataset+':v2')
        self.assertFalse(other.read("anon:a", now=101)["items"])

    def test_merge_transfers_and_removes_guest(self):
        self.store.mutate("anon:a", "anonymous", self.populated)
        self.store.merge("anon:a", "user:u")
        self.assertTrue(self.store.read("user:u")["items"])
        self.assertFalse(self.store.read("anon:a")["items"])

    def test_concurrent_guests_cannot_exceed_limit(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(lambda i:self.store.mutate('anon:'+str(i),'anonymous',self.populated),range(12)))
        with self.database.connect() as db:
            count=db.execute('SELECT count(*) AS n FROM reco.profiles WHERE dataset=%s',(self.dataset,)).fetchone()['n']
        self.assertEqual(count,2)

    def test_concurrent_updates_do_not_lose_state(self):
        from concurrent.futures import ThreadPoolExecutor
        def update(state):
            state['counter']=state.get('counter',0)+1
            return state
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(lambda _:self.store.mutate('anon:a','anonymous',update),range(16)))
        self.assertEqual(self.store.read('anon:a')['counter'],16)

    def test_failed_history_import_rolls_back(self):
        from psycopg.errors import NotNullViolation
        row=dict(user_id='u',item_id='course:1',weight=3,completed=False,owned=True,at=None)
        self.database.replace_history(self.dataset,[row])
        with self.assertRaises(NotNullViolation):
            self.database.replace_history(self.dataset,[dict(row,weight=None)])
        self.assertEqual(self.database.history(self.dataset,'u')[0]['weight'],3)

    def test_background_cleanup_reaches_expired_old_index(self):
        old = ProfileStore(self.database,self.dataset+':old')
        old.mutate('anon:old','anonymous',self.populated,now=100)
        # Проверяем настоящий global DELETE, но откатываем его целиком: тест
        # не должен удалять искусственно просроченные fixtures другого процесса.
        with self.database.connect() as db:
            @contextmanager
            def borrowed_connection():
                yield db
            with db.transaction(force_rollback=True):
                with patch.object(self.database,'connect',borrowed_connection):
                    self.database.cleanup_expired()
                count = db.execute('SELECT count(*) AS n FROM reco.profiles WHERE dataset=%s',(self.dataset+':old',)).fetchone()['n']
                self.assertEqual(count,0)
        # Откат подтверждает, что sweep не фиксировал удаление ни этого, ни чужих seed.
        self.assertTrue(old.read('anon:old',now=109)['items'])


class LLMTests(unittest.TestCase):
    def test_remote_endpoint_is_rejected(self):
        with patch.dict('os.environ', {"LLM_PROVIDER": "ollama", "OLLAMA_URL": "https://example.com", "OLLAMA_MODEL": "test"}):
            self.assertEqual(verify("task", [])[1], "rejected_nonlocal_endpoint")

    def test_bad_llm_json_falls_back(self):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"message":{"content":"invalid"}}'
        with patch.dict('os.environ', {"LLM_PROVIDER": "ollama", "OLLAMA_URL": "http://localhost:11434", "OLLAMA_MODEL": "test"}), patch('urllib.request.urlopen', return_value=response):
            self.assertEqual(verify("task", [])[1], "unavailable_or_invalid_output")


if __name__ == "__main__":
    unittest.main()
