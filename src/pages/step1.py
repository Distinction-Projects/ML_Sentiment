from pathlib import Path
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.ml_sentiment import evaluate_model, preprocess
dash.register_page(__name__, name='1-Model Evaluation', title='Sentiment Analyzer | Model Evaluation')

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load and preprocess data
df = pd.read_csv(DATA_DIR / 'train5.csv')
df.columns = ['Sentiment', 'Text', 'Score']
df['Text'] = df['Text'].astype(str).apply(preprocess)
X = df['Text'].values
y = df['Sentiment'].values
### PAGE LAYOUT ###############################################################################################################
layout = dbc.Container([
    # title
    dbc.Row([
        dbc.Col([html.H3(['Model Evaluation'])], width=12, className='row-titles')
    ]),
    # data input
    dbc.Row([
        dbc.Col([], width = 3),
        dbc.Col([html.P(['Select a model:'], className='par')]),
        dbc.Col([
            dcc.RadioItems(
                options=[
                    {'label': 'Naive Bayes', 'value': 'Naive Bayes'},
                    {'label': 'SVM', 'value': 'SVM'},
                    {'label': 'VADER', 'value': 'VADER'}
                ], 
                value='Naive Bayes', 
                persistence=True, 
                persistence_type='session', 
                id='radio-dataset',
                inline=True
            )
        ], width=6),
        dbc.Col([], width = 1)
    ], className='row-content'),
    # Accuracy metric
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Accuracy", className="text-muted"),
                    html.H3(id='accuracy-display', children='0.00%', className='text-primary')
                ])
            ], className="text-center")
        ], width=3, className="mx-auto")
    ], className='row-content'),
    # Confusion matrix fig
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dcc.Loading(id='p1_1-loading', type='circle', children=dcc.Graph(id='fig-pg1', className='my-graph'))
        ], width = 8),
        dbc.Col([], width = 2)
    ], className='row-content')
    
])
### PAGE CALLBACKS ###############################################################################################################
def create_confusion_matrix_figure(confusion, labels):
    """Create a Plotly heatmap for the confusion matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=confusion,
        x=labels,
        y=labels,
        colorscale='Greens',
        text=confusion,
        texttemplate="%{text}",
        textfont={"size": 16},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=18), x=0.5, xanchor='center'),
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig
# Update fig and accuracy
@callback(
    Output(component_id='fig-pg1', component_property='figure'),
    Output(component_id='accuracy-display', component_property='children'),
    Input(component_id='radio-dataset', component_property='value')
)
def update_evaluation(model_type):
    """Run the model evaluation and return the confusion matrix and accuracy."""
    if model_type == 'VADER':
        # Type 1 is for prebuilt VADER model
        accuracy, precision, recall, confusion, f1 = evaluate_model(X, y, model_type, type=1, k=5)
    else:
        # Type 0 is for custom models
        accuracy, precision, recall, confusion, f1 = evaluate_model(X, y, model_type, type=0, k=5)
    
    labels = ['Negative', 'Neutral', 'Positive']
    fig = create_confusion_matrix_figure(confusion, labels)
    accuracy_text = f"{accuracy:.2%}"
    
    return fig, accuracy_text
