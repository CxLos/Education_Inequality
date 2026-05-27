from unittest.mock import MagicMock, patch

from app.edu_inequality import app


def _get_callback(name):
    for val in app.callback_map.values():
        if 'callback' in val and val['callback'].__name__ == name:
            return getattr(val['callback'], '__wrapped__', val['callback'])
    raise KeyError(f"Callback '{name}' not found in callback_map")


def _mock_client(text="Mock explanation"):
    mock = MagicMock()
    mock.chat.completions.create.return_value.choices[0].message.content = text
    return mock


def test_update_explanation_no_clicks():
    cb = _get_callback('update_explanation')
    result = cb(0, 'Funding Per Student Usd')
    assert result == ("", {}, "")


def test_update_explanation_returns_string_on_click():
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        cb = _get_callback('update_explanation')
        explanation, fig, loading = cb(1, 'Funding Per Student Usd')
    assert isinstance(explanation, str)
    assert loading == ""


def test_update_explanation_returns_figure_on_click():
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        cb = _get_callback('update_explanation')
        explanation, fig, _ = cb(1, 'Funding Per Student Usd')
    assert fig != {}


def test_update_explanation_categorical_column():
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        cb = _get_callback('update_explanation')
        explanation, fig, _ = cb(1, 'School Type')
    assert fig != {}


def test_update_explanation_unknown_column_empty_fig():
    with patch('app.data.ai._get_client', return_value=_mock_client()):
        cb = _get_callback('update_explanation')
        _, fig, _ = cb(1, 'Nonexistent Column')
    assert fig == {}


def test_callback_is_registered():
    names = [
        val['callback'].__name__
        for val in app.callback_map.values()
        if 'callback' in val
    ]
    assert 'update_explanation' in names
