import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app.data import ai as ai_module
from app.data.ai import explain_attribute


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Funding Per Student Usd': [10000.0, 15000.0, 20000.0],
        'Dropout Rate Percent': [5.0, 8.0, 3.0],
        'School Type': ['Public', 'Private', 'Charter'],
    })


def _mock_client(text="Test explanation"):
    mock = MagicMock()
    mock.chat.completions.create.return_value.choices[0].message.content = text
    return mock


def test_explain_attribute_returns_string(sample_df):
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        explanation, _ = explain_attribute(sample_df, 'Funding Per Student Usd')
    assert isinstance(explanation, str)
    assert len(explanation) > 0


def test_explain_attribute_numeric_returns_figure(sample_df):
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        _, fig = explain_attribute(sample_df, 'Funding Per Student Usd')
    assert fig != {}


def test_explain_attribute_categorical_returns_figure(sample_df):
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        _, fig = explain_attribute(sample_df, 'School Type')
    assert fig != {}


def test_explain_attribute_unknown_column_returns_empty_fig(sample_df):
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        _, fig = explain_attribute(sample_df, 'Nonexistent Column')
    assert fig == {}


def test_get_client_initializes_when_none():
    ai_module._client = None
    with patch('app.data.ai.OpenAI') as mock_openai:
        mock_openai.return_value = MagicMock()
        client = ai_module._get_client()
        assert mock_openai.called
        assert client is not None
    ai_module._client = None


def test_get_client_reuses_existing():
    mock = MagicMock()
    ai_module._client = mock
    with patch('app.data.ai.OpenAI') as mock_openai:
        client = ai_module._get_client()
        assert not mock_openai.called
        assert client is mock
    ai_module._client = None
