from pathlib import Path
from dash import Dash, dcc
import dash
import dash_bootstrap_components as dbc

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
PAGES_DIR = BASE_DIR / "pages"

app = Dash(
    __name__,
    use_pages=True,
    assets_folder=str(ASSETS_DIR),
    pages_folder=str(PAGES_DIR),
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
)
server = app.server

############################################################################################
# Import shared components
from src.components.footer import _footer
from src.components.nav import _nav

############################################################################################
# App Layout
app.layout = dbc.Container([
	
	dbc.Row([
        dbc.Col([_nav], width = 2),
        dbc.Col([
            dbc.Row([dash.page_container])
	    ], width = 10),
    ]),
    dbc.Row([
        dbc.Col([], width = 2),
        dbc.Col([
            dbc.Row([_footer])
	    ], width = 10),
    ]),
     dcc.Store(id='browser-memo', data=dict(), storage_type='session')
], fluid=True)

############################################################################################
# Run App
if __name__ == '__main__':
	app.run_server(debug=False, host='0.0.0.0')
