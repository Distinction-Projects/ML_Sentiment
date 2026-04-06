from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html

dash.register_page(__name__, path='/', name='Home', title='NewsLens | Home')

try:
    from src.NewsLens import evaluate_model, preprocess
except ModuleNotFoundError:
    from NewsLens import evaluate_model, preprocess

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train5.csv"

df = pd.read_csv(DATA_PATH)
df.columns = ['Sentiment', 'Text', 'Score']
df['Text'] = df['Text'].astype(str).apply(preprocess)
X = df['Text'].values
y = df['Sentiment'].values

vader_acc, *_ = evaluate_model(X, y, 'VADER', type=1, k=5)
nb_acc, *_ = evaluate_model(X, y, 'Naive Bayes', type=0, k=5)

if nb_acc > vader_acc:
    best_model = 'Naive Bayes'
    diff = nb_acc - vader_acc
elif vader_acc > nb_acc:
    best_model = 'VADER'
    diff = vader_acc - nb_acc
else:
    best_model = 'Tie'
    diff = 0


def make_feature_card(icon_class, title, description, link, link_text):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(html.I(className=icon_class), className='fs-2 text-primary mb-3'),
                html.H5(title, className='card-title text-white'),
                html.P(description, className='card-text text-white'),
                dbc.Button(link_text, href=link, color='outline-primary', size='sm'),
            ],
            className='text-center',
        ),
        className='h-100 shadow-sm',
    )


layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1('NewsLens', className='display-4 fw-bold'),
                                html.P(
                                    'A Dash application for local sentiment-model evaluation and '
                                    'read-only RSS news monitoring. Use the left navigation to move '
                                    'between the model lab and the news workflow dashboards.'
                                ),
                                dbc.Button('Open News Digest', href='/news/digest', color='primary', size='lg', className='me-2'),
                                dbc.Button('Workflow Status', href='/news/workflow-status', outline=True, color='secondary', size='lg'),
                            ],
                            className='text-center py-5',
                        )
                    ],
                    width=12,
                )
            ],
            className='mb-5',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        make_feature_card(
                            'fa-solid fa-newspaper',
                            'News Digest',
                            'Inspect the latest upstream articles, filters, and per-article scoring details.',
                            '/news/digest',
                            'Open Digest',
                        )
                    ],
                    md=6,
                    lg=3,
                    className='mb-4',
                ),
                dbc.Col(
                    [
                        make_feature_card(
                            'fa-solid fa-wave-square',
                            'News Stats',
                            'See source counts, tag distributions, score bins, and trend summaries.',
                            '/news/stats',
                            'Open Stats',
                        )
                    ],
                    md=6,
                    lg=3,
                    className='mb-4',
                ),
                dbc.Col(
                    [
                        make_feature_card(
                            'fa-solid fa-chart-column',
                            'Model Evaluation',
                            'Compare Naive Bayes, SVM, VADER, and precomputed OpenAI labels across corpora.',
                            '/evaluation',
                            'View Metrics',
                        )
                    ],
                    md=6,
                    lg=3,
                    className='mb-4',
                ),
                dbc.Col(
                    [
                        make_feature_card(
                            'fa-solid fa-pen-to-square',
                            'Test Your Text',
                            'Run the local sentiment models on ad hoc text without touching the RSS workflow.',
                            '/text',
                            'Start Testing',
                        )
                    ],
                    md=6,
                    lg=3,
                    className='mb-4',
                ),
            ],
            className='mb-5',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H5('Local Model Snapshot', className='mb-0')),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Span('VADER', className='fw-bold'),
                                                                html.Span(f'{vader_acc:.1%}', className='float-end'),
                                                            ]
                                                        ),
                                                        dbc.Progress(value=vader_acc * 100, color='info', className='mb-3', style={'height': '10px'}),
                                                        html.Div(
                                                            [
                                                                html.Span('Naive Bayes', className='fw-bold'),
                                                                html.Span(f'{nb_acc:.1%}', className='float-end'),
                                                            ]
                                                        ),
                                                        dbc.Progress(value=nb_acc * 100, color='success', className='mb-3', style={'height': '10px'}),
                                                    ],
                                                    md=8,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.P('Best performer on train5', className='text-muted mb-1 small'),
                                                                html.H4(best_model, className='text-primary fw-bold'),
                                                                html.P(f'+{diff:.1%}' if diff else 'No gap', className='text-success'),
                                                            ],
                                                            className='text-center',
                                                        )
                                                    ],
                                                    md=4,
                                                    className='d-flex align-items-center justify-content-center',
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ],
                            className='shadow-sm',
                        )
                    ],
                    lg=8,
                    className='mx-auto',
                )
            ]
        ),
    ],
    fluid=True,
    className='py-4',
)
