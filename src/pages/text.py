"""
OPTIMIZED text.py - Pre-trains models at startup for fast predictions
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from ml_sentiment import preprocess, prebuilt_model

dash.register_page(__name__, path='/text', name='Test Your Text', title='Sentiment Analyzer | Test Your Text')

# ============================================================
# PRE-TRAIN MODELS AT STARTUP (runs once when page loads)
# ============================================================
DATA_LOADED = False
DATA_ERROR = None

# These will hold our pre-trained models
vectorizer = None
nb_model = None
svm_model = None
score_model = None
score_vectorizer = None

try:
    print("[text.py] Loading training data...")
    train_df = pd.read_csv('data/train5.csv')
    train_df.columns = ['Sentiment', 'Text', 'Score']
    train_df['Text'] = train_df['Text'].astype(str).apply(preprocess)
    
    X_train = train_df['Text'].values
    y_train_sentiment = train_df['Sentiment'].values
    y_train_score = pd.to_numeric(train_df['Score'], errors='coerce').fillna(0).values
    
    print(f"[text.py] Loaded {len(X_train)} samples. Pre-training models...")
    
    # Pre-fit the vectorizer ONCE
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Pre-train Naive Bayes
    print("[text.py] Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectorized, y_train_sentiment)
    
    # Pre-train SVM (this is the slow one!)
    print("[text.py] Training SVM (this may take a moment)...")
    svm_model = SVC()
    svm_model.fit(X_train_vectorized, y_train_sentiment)
    
    # Pre-train emotion score model
    print("[text.py] Training emotion score model...")
    score_vectorizer = CountVectorizer()
    X_score_vectorized = score_vectorizer.fit_transform(X_train)
    score_model = LinearRegression()
    score_model.fit(X_score_vectorized, y_train_score)
    
    DATA_LOADED = True
    print("[text.py] All models pre-trained successfully!")
    
except FileNotFoundError:
    DATA_ERROR = "Could not find 'data/train5.csv'. Make sure the file exists."
    print(f"[text.py] ERROR: {DATA_ERROR}")
except Exception as e:
    DATA_ERROR = f"{type(e).__name__}: {str(e)}"
    print(f"[text.py] ERROR: {DATA_ERROR}")


# ============================================================
# LAYOUT
# ============================================================
layout = dbc.Container([
    dbc.Row([
        dbc.Col([html.H3('Test Your Own Text')], width=12, className='row-titles')
    ]),
    
    # Show error if data didn't load
    dbc.Row([
        dbc.Col([
            dbc.Alert(
                [html.I(className="fas fa-exclamation-triangle me-2"), DATA_ERROR],
                color="danger"
            )
        ], width=12)
    ]) if not DATA_LOADED else html.Div(),
    
    dbc.Row([
        dbc.Col([], width=2),
        dbc.Col([
            html.P('Enter your text below and select a model to analyze its sentiment.'),
            dcc.Textarea(
                id='user-text',
                style={
                    'width': '100%', 'height': 120,
                    'backgroundColor': 'white', 'color': '#212529',
                    'border': '1px solid #ced4da', 'padding': '8px', 'borderRadius': '4px'
                },
                placeholder='Type your text here...'
            ),
            html.Br(),
            dcc.RadioItems(
                options=[
                    {'label': html.Span('Naive Bayes', style={'marginRight': '30px'}), 'value': 'Naive Bayes'},
                    {'label': html.Span('SVM', style={'marginRight': '30px'}), 'value': 'SVM'},
                    {'label': html.Span('VADER', style={'marginRight': '30px'}), 'value': 'VADER'}
                ],
                value='Naive Bayes',
                id='model-choice',
                inline=True
            ),
            html.Br(),
            dbc.Button('Analyze', id='analyze-btn', color='primary', disabled=not DATA_LOADED),
            html.Br(), html.Br(),
            html.Div(id='analysis-result')
        ], width=8),
        dbc.Col([], width=2)
    ])
])


# ============================================================
# CALLBACK - Now just uses pre-trained models (fast!)
# ============================================================
@callback(
    Output('analysis-result', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('user-text', 'value'),
    State('model-choice', 'value'),
    prevent_initial_call=True
)
def analyze_text(n_clicks, user_text, model_choice):
    try:
        # Validate
        if not DATA_LOADED:
            return dbc.Alert(f'Training data not available: {DATA_ERROR}', color='danger')
        
        if not user_text or not user_text.strip():
            return dbc.Alert('Please enter some text to analyze.', color='warning')
        
        # Preprocess input
        processed = preprocess(user_text)
        
        if not processed or not processed.strip():
            return dbc.Alert(
                'After removing stopwords, no meaningful words remain. Try a longer sentence.',
                color='warning'
            )
        
        # ---- FAST PREDICTION using pre-trained models ----
        if model_choice == 'VADER':
            pred = prebuilt_model([processed])[0]
            from nltk.sentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(processed)['compound']
        else:
            # Transform input using pre-fitted vectorizer (fast!)
            X_test_vectorized = vectorizer.transform([processed])
            
            # Use pre-trained model (fast!)
            if model_choice == 'Naive Bayes':
                pred = nb_model.predict(X_test_vectorized)[0]
            else:  # SVM
                pred = svm_model.predict(X_test_vectorized)[0]
            
            # Get emotion score using pre-trained model (fast!)
            X_score_vectorized = score_vectorizer.transform([processed])
            score = score_model.predict(X_score_vectorized)[0]
        
        # Format output
        sentiment_map = {'positive': 'Positive', 'neutral': 'Neutral', 'negative': 'Negative'}
        sentiment = sentiment_map.get(str(pred).lower(), str(pred))
        
        # Color based on sentiment
        card_color = 'success' if sentiment == 'Positive' else ('danger' if sentiment == 'Negative' else 'warning')
        
        return dbc.Card([
            dbc.CardBody([
                html.H5('Analysis Result', className='card-title'),
                html.Hr(),
                html.P([
                    html.Strong('Sentiment: '),
                    sentiment
                ], style={'fontSize': '1.2em'}),
                html.P([
                    html.Strong('Emotional Intensity Score: '),
                    f'{float(score):.3f}'
                ], style={'fontSize': '1.1em'}),
                html.Small(f'Model: {model_choice}', className='text-muted')
            ])
        ], color=card_color, outline=True, className='text-center')
    
    except Exception as e:
        import traceback
        print(f"[text.py] Error in callback: {traceback.format_exc()}")
        return dbc.Alert([
            html.Strong(f'Error ({type(e).__name__}): '),
            str(e)
        ], color='danger')
