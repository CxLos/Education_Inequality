import os

COLUMNS = [
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
    'Dropout Rate Percent',
]

DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data',
    'education_inequality_data.csv'
)
