"""Локальные адреса не отправляются в прокси контейнера."""
from pathlib import Path
import re
import unittest
from urllib.request import proxy_bypass_environment


class ProxyBypassTests(unittest.TestCase):
    def test_compose_excludes_local_endpoints_in_both_variable_cases(self):
        compose = (Path(__file__).resolve().parents[1] / 'compose.yaml').read_text()
        for name in ('NO_PROXY', 'no_proxy'):
            value = re.search(r'^\s+' + name + r':\s*(.+)$', compose, re.M).group(1)
            for host in ('localhost:8000', '127.0.0.1:8000', '[::1]:11434',
                         'postgres:5432', 'host.docker.internal:11434'):
                with self.subTest(variable=name, host=host):
                    self.assertTrue(proxy_bypass_environment(host, {'no': value}))
            self.assertFalse(proxy_bypass_environment('openrouter.ai:443', {'no': value}))
