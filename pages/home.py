import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Home', title='Sentiment Analyzer | Home')

layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([
            html.H3('Welcome!'),
            html.P(html.B('App Overview'), className='par')
        ], width=12, className='row-titles')
    ]),
    # Guidelines
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.P([html.B('Hello! This program detects the emotional tone of text.'), html.Br(),
                    'The program will analyze the emotional sentiment of each text sample and classify it as positive, negative, or neutral.'], className='guide'),
            html.P([html.B('1) Model Evaluation'), html.Br(),
                    'View detailed performance metrics including accuracy, precision, recall, and F1 score.'], className='guide'),
            html.P([html.B('2) Model Comparison'), html.Br(),
                    'Compare our model to the pre-built sentiment analyzer VADER.'], className='guide'),
            html.P([html.B('3) Test Your Own Text'), html.Br(),
                    'Enter your own text and see how the program classifies its sentiment.'], className='guide')
        ], width=8),
        dbc.Col([], width=2)
    ])
])
