import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3('Nuorganics Spectroscopy Modeling'),
    dcc.Upload(
        id='upload-training',
        multiple=True,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)