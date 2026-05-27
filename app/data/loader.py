import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go

from app.config import DATA_FILE


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df.rename(columns={col: col.replace('_', ' ').title() for col in df.columns}, inplace=True)
    return df


def build_heatmap(df: pd.DataFrame):
    correlation_matrix = df.select_dtypes(include=np.number).corr()
    heatmap = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        annotation_text=correlation_matrix.round(2).astype(str).values,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Correlation Coefficient'),
        hoverinfo='text',
        text=correlation_matrix.round(2).astype(str).values,
    )
    return heatmap


def build_table(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='center',
            height=30,
            font=dict(size=12),
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left',
            height=25,
            font=dict(size=12),
        ),
    )])
    fig.update_layout(
        margin=dict(l=50, r=50, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig
