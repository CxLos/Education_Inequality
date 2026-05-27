import json

from app.edu_inequality import app
from app.data.loader import load_data


def _find_id(component, target_id):
    """Recursively search a Dash component tree for a component with the given id."""
    if hasattr(component, 'id') and component.id == target_id:
        return True
    children = getattr(component, 'children', None)
    if children is None:
        return False
    if isinstance(children, list):
        return any(_find_id(c, target_id) for c in children)
    return _find_id(children, target_id)


def test_layout_has_attribute_dropdown():
    assert _find_id(app.layout, 'attribute-dropdown')


def test_layout_has_gpt_response():
    assert _find_id(app.layout, 'gpt-response')


def test_layout_has_dropout_graph():
    assert _find_id(app.layout, 'attribute-vs-dropout-graph')


def test_layout_has_explain_button():
    assert _find_id(app.layout, 'explain-button')


def test_data_loads_successfully():
    df = load_data()
    assert len(df) > 0
    assert 'Dropout Rate Percent' in df.columns
