from dash import dcc, html

from app.config import COLUMNS


def create_layout(df_table):
    return html.Div(
        children=[
            # ── Header ──────────────────────────────────────────────────────
            html.Div(
                className='divv',
                children=[
                    html.H1('Education Inequality LLM', className='title'),
                    html.H1('with OpenAI Api', className='title2'),
                    html.Div(
                        className='btn-box',
                        children=[
                            html.A(
                                'Repo',
                                href='https://github.com/CxLos/Education_Inequality',
                                className='repo-btn',
                            )
                        ],
                    ),
                ],
            ),

            # ── Data Table ──────────────────────────────────────────────────
            html.Div(
                className='data-section',
                children=[
                    html.Div(
                        className='data-row',
                        children=[
                            html.Div(
                                className='data-title',
                                children=[
                                    html.H1(
                                        className='table-title',
                                        children='Education Inequality Data Table',
                                    )
                                ],
                            ),
                            html.Div(
                                className='data-table',
                                children=[dcc.Graph(className='data', figure=df_table)],
                            ),
                        ],
                    ),
                ],
            ),

            # ── GPT Section ─────────────────────────────────────────────────
            html.Div(
                className='gpt-section',
                children=[
                    html.Div(children=[]),
                    html.H2("Explore Attribute Impact on Dropout Rate", className='title2'),
                    dcc.Dropdown(
                        className='dropdown',
                        id='attribute-dropdown',
                        options=[{'label': col.replace('_', ' '), 'value': col} for col in COLUMNS],
                        value='',
                        clearable=False,
                    ),
                    html.Button(
                        'Explain Impact',
                        id='explain-button',
                        className='explain',
                        n_clicks=0,
                    ),
                    html.Div(
                        children=[
                            dcc.Loading(
                                className='loading-box',
                                id='loading-spinner',
                                type='circle',
                                children=html.Div(
                                    className='loading',
                                    id='loading-output',
                                ),
                            ),
                        ],
                    ),
                    html.Div(
                        className='gpt-explanation',
                        children=[
                            html.Div(
                                className='gpt-response-box',
                                children=[
                                    html.Div(className='gpt-response', id='gpt-response'),
                                ],
                            ),
                            html.Div(
                                className='gpt-graph-box',
                                children=[
                                    dcc.Graph(
                                        className='gpt-graph',
                                        id='attribute-vs-dropout-graph',
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )
