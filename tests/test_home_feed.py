"""Стартовая страница использует сохранённые события в обоих режимах поиска."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from reco.embeddings import Encoder
from reco.engine import Recommender
from reco.index import fingerprint
from reco.search import SmartSearch
from tests.pg_helpers import PostgresTestCase


class HomeFeedTests(PostgresTestCase):
    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        catalog = []
        for i in range(21):
            ai = i < 12
            text = 'искусственный интеллект автоматизация' if ai else 'налоги учет'
            catalog.append(dict(id='article:'+str(i), kind='article', title=text+' '+str(i),
                description=text, text=text, topics=['ИИ и автоматизация' if ai else 'Налоги'],
                tags=[], eligible=i != 20, history_count=0 if ai else i*10,
                url='/articles/fixture', price=0))
        encoder = Encoder({'backend':'tfidf','min_df':1,'test_nonce':self.dataset})
        vectors = encoder.fit([row['text'] for row in catalog])
        (root/'catalog.json').write_text(json.dumps(catalog))
        encoder.save(root/'encoder.json'); np.save(root/'vectors.npy',vectors)
        self.dataset = fingerprint(root)
        (root/'audit.json').write_text(json.dumps(dict(catalog_fingerprint=self.dataset,catalog_items=20)))
        self.database.replace_history(self.dataset,[])
        self.engine = Recommender(root,database=self.database)
        for i in range(6): self.engine.event('anon:ai','article:'+str(i),'click','click-'+str(i))
        for name in ('reco.search.clarify','reco.engine.verify'):
            guard=patch(name,side_effect=AssertionError('No LLM calls'))
            guard.start(); self.addCleanup(guard.stop)

    def test_reload_uses_history_with_switch_off_or_on_and_eight_one_one(self):
        before = self.engine.store.read('anon:ai')
        for mode in ('normal','smart'):
            result = SmartSearch(self.engine).search(profile_id='anon:ai',search_mode=mode,use_llm=True)
            self.assertTrue(result['feed_mode'])
            self.assertEqual(result['search_mode'],mode)
            self.assertEqual(result['live_items'],6)
            self.assertEqual(len(result['items']),10)
            self.assertEqual(len({row['id'] for row in result['items']}),10)
            sources=[row['feed_source'] for row in result['items']]
            self.assertEqual(sources.count('personal'),8)
            self.assertEqual(sources.count('popular'),1)
            self.assertEqual(sources.count('discovery'),1)
            self.assertTrue(all('ИИ и автоматизация' in row['topics'] for row in result['items'][:8]))
            self.assertIn('Налоги',result['items'][-1]['topics'])
            self.assertIsNone(result['question'])
        self.assertEqual(self.engine.store.read('anon:ai'),before)

    def test_completed_disliked_and_unavailable_never_reappear_in_extra_slots(self):
        self.engine.event('anon:ai','article:19','complete','complete')
        self.engine.event('anon:ai','article:18','dislike','dislike')
        result=SmartSearch(self.engine).search(profile_id='anon:ai',search_mode='normal')
        self.assertFalse({'article:18','article:19','article:20'} & {r['id'] for r in result['items']})

    def test_same_profile_and_day_have_stable_discovery(self):
        first=SmartSearch(self.engine).search(profile_id='anon:ai',search_mode='normal')
        second=SmartSearch(self.engine).search(profile_id='anon:ai',search_mode='normal')
        self.assertEqual([r['id'] for r in first['items']],[r['id'] for r in second['items']])

    def test_fresh_guest_is_marked_starter_without_invented_history(self):
        search=SmartSearch(self.engine)
        with patch.object(self.engine.encoder,'encode',side_effect=AssertionError('No cold-start inference')):
            result=search.search(profile_id='anon:new',search_mode='normal')
        self.assertEqual(result['live_items'],0)
        self.assertEqual(result['profile_topics'],[])
        self.assertEqual(result['match_status'],'starter')
        self.assertEqual([r['feed_source'] for r in result['items'][:8]],['starter']*8)

    def test_latest_interest_gets_more_slots_without_removing_other_topic(self):
        for i in range(12,17): self.engine.event('anon:ai','article:'+str(i),'click','tax-'+str(i))
        result=SmartSearch(self.engine).search(profile_id='anon:ai',search_mode='normal')
        topics=[row['topics'][0] for row in result['items'][:8]]
        self.assertGreater(topics.count('Налоги'),topics.count('ИИ и автоматизация'))
        self.assertGreater(topics.count('ИИ и автоматизация'),0)
        self.assertEqual(topics[0],'Налоги')

    def test_click_refresh_changes_priority_but_duplicate_does_not(self):
        now=time.time()
        profile='user:recency'
        self.engine.event(profile,'article:0','click','old-ai',now=now-3600)
        self.engine.event(profile,'article:12','click','new-tax',now=now-60)
        search=SmartSearch(self.engine)
        def personal_topics():
            result=search.search(profile_id=profile,search_mode='smart')
            return [row['topics'][0] for row in result['items'][:8]]
        first=personal_topics()
        self.assertEqual(first[0],'Налоги')
        self.assertGreater(first.count('Налоги'),first.count('ИИ и автоматизация'))
        self.engine.event(profile,'article:0','click','old-ai',now=now)
        self.assertEqual(personal_topics(),first)
        self.engine.event(profile,'article:0','click','refresh-ai',now=now)
        refreshed=personal_topics()
        self.assertEqual(refreshed[0],'ИИ и автоматизация')
        self.assertGreater(refreshed.count('ИИ и автоматизация'),refreshed.count('Налоги'))
        self.assertGreater(refreshed.count('Налоги'),0)
        self.assertEqual(len(self.engine.store.read(profile)['items']),2)

    def test_positive_interests_halve_per_day_and_future_clock_is_clamped(self):
        now=time.time()
        profile='user:decay'
        self.engine.event(profile,'article:0','engaged','past',now=now-86400)
        self.engine.event(profile,'article:12','click','future',now=now+60)
        with patch('reco.engine.time.time',return_value=now):
            result=self.engine.recommend(profile_id=profile,discovery=True,k=10)
        weights=dict(result['profile_topics'])
        self.assertAlmostEqual(weights['ИИ и автоматизация'],1.5)
        self.assertAlmostEqual(weights['Налоги'],1.0)

    def test_text_normal_search_remains_lexical_without_profile_read(self):
        search=SmartSearch(self.engine)
        with patch.object(self.engine.store,'read',side_effect=AssertionError('No history read')):
            result=search.search(task='налоги',profile_id='anon:ai',search_mode='normal')
        self.assertFalse(result.get('feed_mode',False))
        self.assertTrue(all('Налоги' in row['topics'] for row in result['items']))

    def test_no_popularity_data_does_not_claim_a_material_is_popular(self):
        self.engine.popularity[:] = 0
        result=SmartSearch(self.engine).search(profile_id='anon:ai',search_mode='normal')
        self.assertFalse(any(row['feed_source']=='popular' for row in result['items']))
        self.assertNotIn('популярное',result['summary'])

    def test_tiny_or_empty_catalog_and_invalid_discovery(self):
        for count in (3,0):
            for i,row in enumerate(self.engine.catalog): row['eligible']=i<count
            result=SmartSearch(self.engine).search(profile_id='anon:ai',search_mode='normal')
            self.assertEqual(len(result['items']),count)
            self.assertTrue(all(r['feed_source']=='personal' for r in result['items']))
        for options in ({'discovery':'yes'},{'discovery':True,'task':'ИИ'},
                        {'discovery':True,'use_llm':True}):
            with self.assertRaises(ValueError): self.engine.recommend(**options)


class FeedCompositionTests(unittest.TestCase):
    def test_equal_recency_preserves_balance_and_old_weights_do_not_dominate(self):
        from reco.feed import compose_feed
        rows=[{'id':str(i),'topics':['AI' if i<8 else 'Tax']} for i in range(16)]
        args=(rows,list(range(16)),list(range(16)),10,np.zeros(16),{'0':6,'8':5},'fixture')
        equal,_=compose_feed(*args,last_activity={'AI':100,'Tax':100})
        self.assertEqual(sum(i<8 for i in equal[:8]),4)
        recent,_=compose_feed(*args,last_activity={'AI':99,'Tax':100})
        self.assertEqual(recent[0],8)
        self.assertGreater(sum(i>=8 for i in recent[:8]),4)
        self.assertGreater(sum(i<8 for i in recent[:8]),0)
        self.assertEqual(len(set(recent)),10)

    def test_discovery_uses_all_four_known_interests(self):
        from reco.feed import compose_feed
        rows=[{'id':str(i),'topics':[topic]} for i,topic in enumerate('AAABBBCCCDDDEEE')]
        popularity=np.zeros(15); popularity[2]=1; popularity[10]=0.5
        seen={'0':4,'3':3,'6':2,'9':1}
        selected,sources=compose_feed(rows,list(range(15)),list(range(15)),10,popularity,seen,'3')
        discovery=next(i for i in selected if sources[i]=='discovery')
        self.assertEqual(rows[discovery]['topics'],['E'])

    def test_secondary_topics_participate_without_duplicate_cards(self):
        from reco.feed import compose_feed
        rows=[{'id':str(i),'topics':['Налоги','Команда']} for i in range(5)] + [
            {'id':str(i),'topics':['ИИ']} for i in range(5,10)]
        seen={'0':1,'5':1}
        a,sources=compose_feed(rows,list(range(10)),list(range(10)),10,np.ones(10),seen,'fixture')
        self.assertEqual(len(set(a)),10)
        # Перестановка меток не меняет предпочтения и не превращает «Команду» в налоги.
        for row in rows: row['topics']=list(reversed(row['topics']))
        b,_=compose_feed(rows,list(range(10)),list(range(10)),10,np.ones(10),seen,'fixture')
        self.assertEqual(a,b)
