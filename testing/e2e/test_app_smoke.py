import threading
import time

import pytest
import requests

from app.edu_inequality import app, server

PORT = 8152


@pytest.fixture(scope='module', autouse=True)
def flask_server():
    t = threading.Thread(
        target=lambda: server.run(host='127.0.0.1', port=PORT),
        daemon=True,
    )
    t.start()
    time.sleep(2)
    yield


def test_homepage_returns_200():
    r = requests.get(f'http://127.0.0.1:{PORT}/')
    assert r.status_code == 200


def test_homepage_contains_dash_content():
    r = requests.get(f'http://127.0.0.1:{PORT}/')
    assert '_dash' in r.text or 'dash' in r.text.lower()


def test_css_is_served():
    r = requests.get(f'http://127.0.0.1:{PORT}/')
    assert r.status_code == 200
