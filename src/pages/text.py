import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from ml_sentiment import preprocess, evaluate_model, my_model, prebuilt_model, emotion_score

dash.register_page(__name__, path='/text', name='Test Your Text', title='Sentiment Analyzer | Test Your Text')

# Load and preprocess training data for model fitting
train_df = pd.read_csv('data/train5.csv')
train_df.columns = ['Sentiment', 'Text', 'Score']
train_df['Text'] = train_df['Text'].astype(str).apply(preprocess)
X_train = train_df['Text'].values
y_train = train_df['Sentiment'].values

layout = dbc.Container([
    dbc.Row([
        dbc.Col([html.H3('Test Your Own Text')], width=12, className='row-titles')
    ]),
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.P('Enter your text below and select a model to analyze its sentiment.'),
            dcc.Textarea(id='user-text', style={'width': '100%', 'height': 100}, placeholder='Type your text here...'),
            html.Br(),
            dcc.RadioItems(
                options=[
                    {'label': html.Span('Naive Bayes', style={'margin-right': '30px'}), 'value': 'Naive Bayes'},
                    {'label': html.Span('SVM', style={'margin-right': '30px'}), 'value': 'SVM'},
                    {'label': html.Span('VADER', style={'margin-right': '30px'}), 'value': 'VADER'}
                ],
                value='Naive Bayes',
                id='model-choice',
                inline=True
            ),
            html.Br(),
            dbc.Button('Analyze', id='analyze-btn', color='primary'),
            html.Br(), html.Br(),
            html.Div(id='analysis-result')
        ], width=8),
        dbc.Col([], width=2)
    ])
])

@callback(
    Output('analysis-result', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('user-text', 'value'),
    State('model-choice', 'value'),
    prevent_initial_call=True
)
def analyze_text(n_clicks, user_text, model_choice):
    if not user_text or not user_text.strip():
        return dbc.Alert('Please enter some text to analyze.', color='warning')
    # Preprocess input
    processed = preprocess(user_text)
    # Model prediction
    if model_choice == 'VADER':
        pred = prebuilt_model([processed])[0]
        # Use VADER's compound score for intensity
        from nltk.sentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(processed)['compound']
    else:
        pred = my_model(X_train, y_train, [processed], model_choice)[0]
        # Use emotion_score regression for intensity
        score = emotion_score(X_train, y_train, [processed])[0]
    # Format output
    sentiment_map = {'positive': 'Positive', 'neutral': 'Neutral', 'negative': 'Negative'}
    sentiment = sentiment_map.get(pred.lower(), pred)
    return dbc.Card([
        dbc.CardBody([
            html.H5('Analysis Result'),
            html.P(f'Sentiment: {sentiment}', style={'font-size': '1.2em'}),
            html.P(f'Emotional Intensity Score: {score:.3f}', style={'font-size': '1.1em'}),
        ])
    ], color='info', className='text-center')
