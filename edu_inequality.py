# =================================== IMPORTS ================================= #

# ------ System Imports ------ #
import os
import sys

# ------ Python Imports ------ #
# import csv, sqlite3
import numpy as np 
import pandas as pd 
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter

# ------ OpenAI ------ #
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ------ Machine Learning Imports ------ #
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.pipeline import Pipeline

# ------ Dash Imports ------ #
import dash
from dash import dcc, html
# from dash.dependencies import Input, Output, State

# -------------------------------------- DATA ------------------------------------------- #

current_dir = os.getcwd()
current_file = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Current Directory: {current_dir}")
# print(f"Current File: {current_file}")
# print(f"Script Directory: {script_dir}")
# con = sqlite3.connect("Chicago_Schools.db")
# cur = con.cursor()

# file = r'c:\Users\CxLos\OneDrive\Documents\Portfolio Projects\Machine Learning\Kidney_Disease_Outcome\data\kidney_disease_dataset.xlsx'

file = os.path.join(script_dir, 'data', 'education_inequality_data.csv')

df = pd.read_csv(file)

# print(df.head(10))
# print(f'DF Shape: \n {df.shape}')
# print(f'Number of rows: {df.shape[0]}')
# print(f'Column names: \n {df.columns}')
# print(df.info())
# print(df.describe())
# print(df.dtypes)

# =========================== OpenAI ========================== #

# attribute = 'CPU frequency'  # Example attribute to analyze
attribute = ''  # Example attribute to analyze

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    # model="gpt-4",
    model="gpt-3.5-turbo",  # Use this if you don't have GPT-4 access
    messages=[
        {"role": "system", # Tell the model what it is
         "content": "You're a data analyst."}, # This is the system message
        {"role": "user", # User's message/ role
         # Now we ask the model to explain the impact of the attribute on the dropout rate
         "content": f"Explain how {attribute} affects High School drop out rate."} 
    ]
)


# print(response.choices[0].message.content)

# ================================= Columns ================================= #

columns = [
    # 'id', 
    # 'school_name', 
    # 'state', 
    # 'school_type',
    # 'grade_level',
    # 'funding_per_student_usd',
    # 'avg_test_score_percent',
    # 'student_teacher_ratio',
    # 'percent_low_income',
    # 'percent_minority',
    # 'internet_access_percent', 
    # 'dropout_rate_percent'
    # --------------------------
    'Id',
    'School Name',
    'State',
    'School Type',
    'Grade Level',
    'Funding Per Student Usd', 
    'Avg Test Score Percent',
    'Student Teacher Ratio',
    'Percent Low Income',
    'Percent Minority',
    'Internet Access Percent', 
    'Dropout Rate Percent'
]

# Rename columns to remove underscores and title case them
df.rename(columns={col: col.replace('_', ' ').title() for col in df.columns}, inplace=True)

# print(f'Column names after: \n {df.columns}')

# ============================== Data Preprocessing ========================== #

# Missing Values
missing = df.isnull().sum()
# print('Columns with missing values before fillna: \n', missing[missing > 0])

# # ========================== DataFrame Table ========================== #

df_table = go.Figure(data=[go.Table(
    # columnwidth=[50, 50, 50],  # Adjust the width of the columns
    header=dict(
        values=list(df.columns),
        fill_color='paleturquoise',
        align='center',
        height=30,  # Adjust the height of the header cells
        # line=dict(color='black', width=1),  # Add border to header cells
        font=dict(size=12)  # Adjust font size
    ),
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='lavender',
        align='left',
        height=25,  # Adjust the height of the cells
        # line=dict(color='black', width=1),  # Add border to cells
        font=dict(size=12)  # Adjust font size
    )
)])

df_table.update_layout(
    margin=dict(l=50, r=50, t=30, b=40),  # Remove margins
    height=400,
    # width=1500,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# ============================== Dash Application ========================== #

# ...existing code...

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    children=[
        html.Div(
            className='divv',
            children=[
                html.H1('Education Inequality LLM', className='title'),
                html.Div(
                    className='btn-box',
                    children=[
                        html.A(
                            'Repo',
                            href='https://github.com/CxLos/Education_Inequality',
                            className='btn'
                        )
                    ]
                )
            ]
        ),
        
        # ----------------------- Data Table ----------------------- #
        
        html.Div(
            className='row0',
            children=[
                html.Div(
                    className='table',
                    children=[
                        html.H1(
                            className='table-title',
                            children='Education Inequality Data Table'
                        )
                    ]
                ),
                html.Div(
                    className='table2',
                    children=[
                        dcc.Graph(
                            className='data',
                            figure=df_table
                        )
                    ]
                )
            ]
        ),
        
        # ----------------------- Graphs ----------------------- #
        
        # html.Div(
        #     className='row1',
        #     children=[
        #         html.Div(
        #             className='graph1',
        #             children=[
        #                 dcc.Graph()
        #             ]
        #         ),
        #         html.Div(
        #             className='graph2',
        #             children=[
        #                 dcc.Graph()
        #             ]
        #         )
        #     ]
        # ),
        
        # ------------------------ GPT-Graph ----------------------- #
        
        html.Div(
            className='gpt-section',
            children=[
                html.H2("Explore Attribute Impact on Dropout Rate", className='title2'),
                dcc.Dropdown(
                    className='dropdown',
                    id='attribute-dropdown',
                    options=[{'label': col.replace('_', ' '), 'value': col} for col in columns],
                    value='',
                    clearable=False
                ),
                
                html.Button('Explain Impact', 
                            id='explain-button',
                            className='explain',
                            n_clicks=0),
                
                html.Div(
                    className='graph33', 
                    id='gpt-response', 
                    style={'whiteSpace': 'pre-line', 'marginTop': '20px'}
                    ),
                dcc.Graph(
                    className='gpt-graph',
                    id='attribute-vs-price-graph', 
                    # style={'marginTop': '40px'}
                    ),
            ]
        ),
        
        # ----------------------- README ----------------------- #
html.Div(
    className='readme-section',
    children=[
        html.H2("📘 README"),

        html.H4("Description"),
        html.P(
            "This project leverages machine learning to predict the likelihood of chronic kidney disease (CKD) using a clinical dataset. "
            "The goal is to assist in early detection of CKD by analyzing relevant patient biomarkers and visualizing key insights through an "
            "interactive Plotly/Dash dashboard. The project includes preprocessing, training a model, evaluating performance, and highlighting "
            "feature importance."
        ),

        html.H4("📦 Installation"),
        html.P("To run this project locally, follow these steps:"),
        html.Pre([
            html.Code(
                "git clone https://github.com/CxLos/Education Inequality\n"
                "cd Kidney_Disease_Outcome\n"
                "pip install -r requirements.txt"
            )
        ]),

        html.H4("🧪 Methodology"),
        html.Ul([
            html.Li("Dataset sourced from Kaggle with 2,300+ patients' clinical measurements."),
            html.Li("Preprocessing included handling missing values, outlier treatment, categorical encoding, and normalization."),
            html.Li("Models trained: Logistic Regression, Decision Tree, and Random Forest."),
            html.Li("Evaluated using accuracy, precision, recall, F1-score."),
            html.Li("Feature importance used to understand drivers of CKD prediction.")
        ]),

        html.H4("🔍 Insights"),
        html.Ul([
            html.Li("Random Forest achieved the highest overall performance in accuracy and F1 score, indicating a strong balance between precision and recall."),
            html.Li("Decision Tree showed decent performance but slightly lagged behind Random Forest."),
            html.Li("Logistic Regression had the lowest scores across most metrics, making it the least effective model in this comparison.")
        ]),

        html.H4("✅ Conclusion"),
        html.P(
            "This project demonstrates the application of machine learning for health diagnostics. "
            "By combining statistical insights with interactive visualizations, it offers a powerful tool for analyzing kidney disease outcomes. "
            "Future improvements could include using ensemble models or deploying the app with live patient data integration."
        ),

        html.H4("📄 License"),
        html.P("MIT License © 2025 CxLos"),
html.Code(
    "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. \n\n"
    "THE SOFTWARE IS PROVIDED \"AS IS\""
    "WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."
)
    ]
)

    ]
)

# --------------- Callback ----------------- #

@app.callback(
    [dash.dependencies.Output('gpt-response', 'children'),
     dash.dependencies.Output('attribute-vs-price-graph', 'figure')],
    [dash.dependencies.Input('explain-button', 'n_clicks')],
    [dash.dependencies.State('attribute-dropdown', 'value')]
)

def update_explanation(n_clicks, selected_attribute):
    if n_clicks == 0:
        return "", {}

    # Ask the LLM how the selected attribute affects dropout rate
    messages = [
        {"role": "system", "content": "You're a data analyst."},
        {"role": "user", "content": f"Explain how {selected_attribute} affects high school dropout rates."}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    explanation = response.choices[0].message.content

    # Assume df has columns like 'Dropout Rate' and selected_attribute
    if selected_attribute in df.columns and 'Dropout Rate Percent' in df.columns:
        filtered = df[[selected_attribute, 'Dropout Rate Percent']].dropna()

        if pd.api.types.is_numeric_dtype(filtered[selected_attribute]):
            fig = px.scatter(
                filtered,
                x=selected_attribute,
                y='Dropout Rate Percent',
                title=f"{selected_attribute.replace('_', ' ')} vs Dropout Rate Percent",
                labels={
                    selected_attribute: selected_attribute.replace('_', ' '),
                    'Dropout Rate Percent': 'High School Dropout Rate'
                }
            ).update_layout(
                height=700,
                # width=800,
            )
        else:
            fig = px.box(
                filtered,
                x=selected_attribute,
                y='Dropout Rate Percent',
                title=f"{selected_attribute.replace('_', ' ')} vs Dropout Rate Percent",
                labels={
                    selected_attribute: selected_attribute.replace('_', ' '),
                    'Dropout Rate Percent': 'High School Dropout Rate'
                }
            )
    else:
        fig = {}

    return explanation, fig


# ---------------------- End ------------------------- #

print(f"Serving Flask app '{current_file}'! 🚀")

if __name__ == '__main__':
    app.run_server(debug=
                    True)
                    # False)
# =================================== Updated Database ================================= #

# updated_path = f'data/kidney_disease_outcome_cleaned.xlsx'.xlsx'
# data_path = os.path.join(script_dir, updated_path)

# with pd.ExcelWriter(data_path, engine='xlsxwriter') as writer:
#     df.to_excel(
#             writer, 
#             sheet_name=f'Engagement {current_month} {report_year}', 
#             startrow=1, 
#             index=False
#         )

#     # Access the workbook and each worksheet
#     workbook = writer.book
#     sheet1 = writer.sheets['Kidney Disease Outcome']
    
#     # Define the header format
#     header_format = workbook.add_format({
#         'bold': True, 
#         'font_size': 13, 
#         'align': 'center', 
#         'valign': 'vcenter',
#         'border': 1, 
#         'font_color': 'black', 
#         'bg_color': '#B7B7B7',
#     })
    
#     # Set column A (Name) to be left-aligned, and B-E to be right-aligned
#     left_align_format = workbook.add_format({
#         'align': 'left',  # Left-align for column A
#         'valign': 'vcenter',  # Vertically center
#         'border': 0  # No border for individual cells
#     })

#     right_align_format = workbook.add_format({
#         'align': 'right',  # Right-align for columns B-E
#         'valign': 'vcenter',  # Vertically center
#         'border': 0  # No border for individual cells
#     })
    
#     # Create border around the entire table
#     border_format = workbook.add_format({
#         'border': 1,  # Add border to all sides
#         'border_color': 'black',  # Set border color to black
#         'align': 'center',  # Center-align text
#         'valign': 'vcenter',  # Vertically center text
#         'font_size': 12,  # Set font size
#         'font_color': 'black',  # Set font color to black
#         'bg_color': '#FFFFFF'  # Set background color to white
#     })

#     # Merge and format the first row (A1:E1) for each sheet
#     sheet1.merge_range('A1:N1', f'Engagement Report {current_month} {report_year}', header_format)

#     # Set column alignment and width
#     # sheet1.set_column('A:A', 20, left_align_format)   

#     print(f"Kidney Disease Excel file saved to {data_path}")

# -------------------------------------------- KILL PORT ---------------------------------------------------

# netstat -ano | findstr :8050
# taskkill /PID 24772 /F
# npx kill-port 8050

# ---------------------------------------------- Host Application -------------------------------------------

# 1. pip freeze > requirements.txt
# 2. add this to procfile: 'web: gunicorn kidney_disease:server'
# 3. heroku login
# 4. heroku create
# 5. git push heroku main

# Create venv 
# virtualenv venv 
# source venv/bin/activate # uses the virtualenv

# Update PIP Setup Tools:
# pip install --upgrade pip setuptools

# Install all dependencies in the requirements file:
# pip install -r requirements.txt

# Check dependency tree:
# pipdeptree
# pip show package-name

# Remove:
# pypiwin32
# pywin32
# jupytercore
# ipykernel
# ipython

# Add:
# gunicorn==22.0.0

# ----------------------------------------------------

# Name must start with a letter, end with a letter or digit and can only contain lowercase letters, digits, and dashes.

# Heroku Setup:
# heroku login
# heroku create kidney-disease-outcome
# heroku git:remote -a kidney-disease-outcome
# git push heroku main

# Clear Heroku Cache:
# heroku plugins:install heroku-repo
# heroku repo:purge_cache -a mc-impact-11-2024

# Set buildpack for heroku
# heroku buildpacks:set heroku/python

# Heatmap Colorscale colors -----------------------------------------------------------------------------

#   ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
            #  'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
            #  'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
            #  'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
            #  'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
            #  'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
            #  'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
            #  'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
            #  'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
            #  'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
            #  'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
            #  'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
            #  'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
            #  'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
            #  'ylorrd'].

# rm -rf ~$bmhc_data_2024_cleaned.xlsx
# rm -rf ~$bmhc_data_2024.xlsx
# rm -rf ~$bmhc_q4_2024_cleaned2.xlsx