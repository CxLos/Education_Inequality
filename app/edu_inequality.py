import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import dash

from app.data.loader import load_data, build_table
from app.layouts.layout import create_layout
from app.callbacks.edu_callbacks import register_callbacks

assets_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'assets'
)

app = dash.Dash(__name__, assets_folder=assets_folder)
server = app.server

df = load_data()
df_table = build_table(df)

app.layout = create_layout(df_table)
register_callbacks(app, df)

if __name__ == '__main__':  # pragma: no cover
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=True)
