"""
DIAGNOSTIC VERSION - Use this to find the issue
Replace your text.py with this file temporarily
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import traceback

dash.register_page(__name__, path='/text', name='Test Your Text', title='Sentiment Analyzer | Test Your Text')

# ============ STEP 1: Test imports ============
import_errors = []

try:
    import numpy as np
    import pandas as pd
except Exception as e:
    import_errors.append(f"numpy/pandas: {e}")

try:
    from ml_sentiment import preprocess, my_model, prebuilt_model, emotion_score
except Exception as e:
    import_errors.append(f"ml_sentiment: {e}")

# ============ STEP 2: Test data loading ============
data_status = None
X_train = None
y_train_sentiment = None
y_train_score = None

try:
    train_df = pd.read_csv('data/train5.csv')
    train_df.columns = ['Sentiment', 'Text', 'Score']
    train_df['Text'] = train_df['Text'].astype(str).apply(preprocess)
    X_train = train_df['Text'].values
    y_train_sentiment = train_df['Sentiment'].values
    y_train_score = pd.to_numeric(train_df['Score'], errors='coerce').fillna(0).values
    data_status = f"✅ Data loaded: {len(X_train)} samples"
except FileNotFoundError as e:
    data_status = f"❌ File not found: {e}"
except Exception as e:
    data_status = f"❌ Data error: {type(e).__name__}: {e}"

# ============ LAYOUT ============
layout = dbc.Container([
    dbc.Row([
        dbc.Col([html.H3('Test Your Own Text - DIAGNOSTIC MODE')], width=12)
    ]),
    
    # Diagnostic Panel
    dbc.Card([
        dbc.CardHeader("🔍 Diagnostic Information"),
        dbc.CardBody([
            html.H6("Import Status:"),
            html.Pre("✅ All imports OK" if not import_errors else "\n".join(import_errors)),
            html.Hr(),
            html.H6("Data Loading Status:"),
            html.Pre(data_status),
            html.Hr(),
            html.H6("Training Data Sample (first 3):"),
            html.Pre(
                str(X_train[:3]) if X_train is not None else "No data loaded"
            ),
        ])
    ], className="mb-4", color="light"),
    
    dbc.Row([
        dbc.Col([
            html.P('Enter your text below:'),
            dcc.Textarea(
                id='user-text',
                style={'width': '100%', 'height': 100},
                placeholder='Type your text here...',
                value='I love this product!'  # Default value for testing
            ),
            html.Br(),
            dcc.RadioItems(
                options=[
                    {'label': ' Naive Bayes ', 'value': 'Naive Bayes'},
                    {'label': ' SVM ', 'value': 'SVM'},
                    {'label': ' VADER ', 'value': 'VADER'}
                ],
                value='Naive Bayes',
                id='model-choice',
                inline=True
            ),
            html.Br(),
            dbc.Button('Analyze', id='analyze-btn', color='primary', n_clicks=0),
            html.Br(), html.Br(),
            
            # Debug output
            html.Div(id='debug-output', style={'backgroundColor': '#f8f9fa', 'padding': '10px'}),
            html.Br(),
            html.Div(id='analysis-result')
        ], width=10)
    ])
])

@callback(
    [Output('analysis-result', 'children'),
     Output('debug-output', 'children')],
    Input('analyze-btn', 'n_clicks'),
    State('user-text', 'value'),
    State('model-choice', 'value'),
    prevent_initial_call=True
)
def analyze_text(n_clicks, user_text, model_choice):
    debug_log = []
    debug_log.append(f"🔹 Callback triggered! n_clicks={n_clicks}")
    debug_log.append(f"🔹 model_choice='{model_choice}'")
    debug_log.append(f"🔹 user_text='{user_text[:50] if user_text else None}...'")
    
    try:
        # Check data
        if X_train is None:
            debug_log.append("❌ X_train is None - data not loaded!")
            return (
                dbc.Alert("Data not loaded!", color="danger"),
                html.Pre("\n".join(debug_log), style={'fontSize': '0.85em'})
            )
        
        debug_log.append(f"🔹 X_train has {len(X_train)} samples")
        
        # Validate input
        if not user_text or not user_text.strip():
            debug_log.append("❌ Empty input text")
            return (
                dbc.Alert('Please enter some text.', color='warning'),
                html.Pre("\n".join(debug_log), style={'fontSize': '0.85em'})
            )
        
        # Preprocess
        debug_log.append("🔹 Preprocessing text...")
        processed = preprocess(user_text)
        debug_log.append(f"🔹 Processed: '{processed}'")
        
        if not processed.strip():
            debug_log.append("❌ Processed text is empty!")
            return (
                dbc.Alert('Text became empty after preprocessing', color='warning'),
                html.Pre("\n".join(debug_log), style={'fontSize': '0.85em'})
            )
        
        # Predict
        if model_choice == 'VADER':
            debug_log.append("🔹 Using VADER model...")
            pred = prebuilt_model([processed])[0]
            from nltk.sentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(processed)['compound']
        else:
            debug_log.append(f"🔹 Using {model_choice} model...")
            debug_log.append(f"🔹 Calling my_model(X_train[{len(X_train)}], y_train[{len(y_train_sentiment)}], [processed], '{model_choice}')")
            
            pred = my_model(X_train, y_train_sentiment, [processed], model_choice)[0]
            debug_log.append(f"🔹 Prediction returned: '{pred}'")
            
            score = emotion_score(X_train, y_train_score, [processed])[0]
            debug_log.append(f"🔹 Score returned: {score}")
        
        debug_log.append("✅ SUCCESS!")
        
        # Format result
        sentiment_map = {'positive': 'Positive', 'neutral': 'Neutral', 'negative': 'Negative'}
        sentiment = sentiment_map.get(str(pred).lower(), str(pred))
        
        result = dbc.Card([
            dbc.CardBody([
                html.H5('Analysis Result'),
                html.P(f'Sentiment: {sentiment}', style={'fontSize': '1.2em'}),
                html.P(f'Score: {score:.3f}', style={'fontSize': '1.1em'}),
            ])
        ], color='success', outline=True)
        
        return (
            result,
            html.Pre("\n".join(debug_log), style={'fontSize': '0.85em', 'color': 'green'})
        )
        
    except Exception as e:
        debug_log.append(f"❌ EXCEPTION: {type(e).__name__}: {e}")
        debug_log.append(f"❌ Traceback:\n{traceback.format_exc()}")
        
        return (
            dbc.Alert([
                html.Strong(f"Error: {type(e).__name__}"),
                html.P(str(e))
            ], color="danger"),
            html.Pre("\n".join(debug_log), style={'fontSize': '0.85em', 'color': 'red'})
        )
