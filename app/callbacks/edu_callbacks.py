import dash

from app.data.ai import explain_attribute


def register_callbacks(app, df):

    @app.callback(
        [
            dash.dependencies.Output('gpt-response', 'children'),
            dash.dependencies.Output('attribute-vs-dropout-graph', 'figure'),
            dash.dependencies.Output('loading-output', 'children'),
        ],
        [dash.dependencies.Input('explain-button', 'n_clicks')],
        [dash.dependencies.State('attribute-dropdown', 'value')],
    )
    def update_explanation(n_clicks, selected_attribute):
        if n_clicks == 0:
            return "", {}, ""
        explanation, fig = explain_attribute(df, selected_attribute)
        return explanation, fig, ""
