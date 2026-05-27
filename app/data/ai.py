import os

import pandas as pd
import plotly.express as px
from openai import OpenAI

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def explain_attribute(df: pd.DataFrame, selected_attribute: str):
    """Call GPT-3.5-turbo and return (explanation_text, scatter_or_box_figure)."""
    messages = [
        {"role": "system", "content": "You're a data analyst."},
        {"role": "user", "content": f"Explain how {selected_attribute} affects high school dropout rates."},
    ]
    response = _get_client().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    explanation = response.choices[0].message.content

    if selected_attribute in df.columns and 'Dropout Rate Percent' in df.columns:
        filtered = df[[selected_attribute, 'Dropout Rate Percent']].dropna()
        label = selected_attribute.replace('_', ' ')
        if pd.api.types.is_numeric_dtype(filtered[selected_attribute]):
            fig = px.scatter(
                filtered,
                x=selected_attribute,
                y='Dropout Rate Percent',
                title=f"{label} vs Dropout Rate Percent",
                labels={selected_attribute: label, 'Dropout Rate Percent': 'High School Dropout Rate'},
            ).update_layout(title_x=0.5).update_traces(
                hovertemplate=f"{label}: <b>%{{x}}</b><br>Dropout Rate: <b>%{{y:.2f}}%</b><extra></extra>"
            )
        else:
            fig = px.box(
                filtered,
                x=selected_attribute,
                y='Dropout Rate Percent',
                title=f"{label} vs Dropout Rate Percent",
                labels={selected_attribute: label, 'Dropout Rate Percent': 'High School Dropout Rate'},
            ).update_layout(title_x=0.5).update_traces(
                hovertemplate=f"{label}: <b>%{{x}}</b><br>Dropout Rate: <b>%{{y:.2%}}</b><extra></extra>"
            )
    else:
        fig = {}

    return explanation, fig
