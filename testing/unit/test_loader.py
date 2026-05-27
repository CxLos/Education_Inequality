import pandas as pd
import plotly.graph_objects as go

from app.data.loader import load_data, build_heatmap, build_table


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)


def test_load_data_not_empty():
    df = load_data()
    assert len(df) > 0


def test_load_data_has_expected_columns():
    df = load_data()
    assert 'Dropout Rate Percent' in df.columns
    assert 'Funding Per Student Usd' in df.columns
    assert 'School Type' in df.columns


def test_load_data_renames_columns():
    df = load_data()
    # Raw CSV uses snake_case; loader should title-case them
    assert 'dropout_rate_percent' not in df.columns


def test_build_heatmap_returns_figure():
    df = load_data()
    fig = build_heatmap(df)
    assert fig is not None


def test_build_table_returns_figure():
    df = load_data()
    fig = build_table(df)
    assert isinstance(fig, go.Figure)


def test_build_table_has_table_trace():
    df = load_data()
    fig = build_table(df)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Table)


def test_build_table_columns_match_df():
    df = load_data()
    fig = build_table(df)
    assert list(fig.data[0].header.values) == list(df.columns)
