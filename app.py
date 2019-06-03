import numpy as np
import pandas as pd

import dash
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

from Model import Model
from AppLogic import GraphLogic, DataLogic
from dash.dependencies import Input, Output

# http://127.0.0.1:8050/

#-#-#-#-# MAIN BEFORE UI #-#-#-#-#
# ------ Graph Logic ------
logic = GraphLogic()
logic.load_w2v("Word2vec/word2vec_twitter")

word_vectors = logic.create_word_vectors(100)
word_vocab = logic.create_word_vocab(100)

logic.create_tsne()
x_w2v, y_w2v, z_w2v = logic.get_dimensions()

# ----- Import and clean CSV -----
data_logic = DataLogic()

X_test, Y_test, test_df = data_logic.text_preproc()
dropdown_options = data_logic.get_dropdown()

# -------- LOAD MACHINE LEARNING MODEL ---------
model = Model(load = True, model_name = "TrainedModelGloVe")


predictions = model.make_prediction(X_test)
aciertos = 0
for i in range(0, len(predictions)):    
    if np.argmax(predictions[i]) == (Y_test[i] // 4):
        aciertos += 1

print("Aciertos: ", aciertos, " sobre ", len(predictions), " Porcentaje: ", aciertos / len(predictions))
"""
import sys
sys.exit("")
""" 

# # # # # # # # # # # # # 
# # # # UI CODE # # # # #
# # # # # # # # # # # # # 
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#F4F4F8',
    'text': '#323232'
}

# # # # TEXT STYLES # # # # #
# # # # # # # # # # # # # # # 
text_style ={
            'color': colors['text'],
            'font-size': '4.0rem',
            'color': '#4D637F',
            'font-family': 'Dosis'}

text_style_p ={
            'color': colors['text'],
            'font-size': '1.7rem',
            'color': '#2b3949',
            'font-family': 'Dosis'}

text_style_h4 ={
            'color': colors['text'],
            'font-size': '3.0rem',
            'color': '#4D637F',
            'font-family': 'Dosis'}

# # # # LAYOUT # # # # #
# # # # # # # # # # # # #
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    #-#-#-#-#-# INTRODUCTION #-#-#-#-#-#
    html.Div([
        html.H1("Análisis de Sentimientos"),
        html.H5("Alejandro Pérez Sanjuán")
        ], style={
            'marginLeft': 600,
            'marginRight': 600,
            'color': colors['text'],
            'font-size': '4.0rem',
            'color': '#4D637F',
            'font-family': 'Dosis',}
    ),

    html.Hr([], style={'marginLeft': 600, 'marginRight': 600}),

    # Buttons: https://community.plot.ly/t/input-two-or-more-button-how-to-tell-which-button-is-pressed/5788/29

    #-#-#-#-#-#  TWEET INPUTS  #-#-#-#-#-#
    html.Div([
        html.H4(["Predecir la polaridad de un tweet"],
                style=text_style_h4),
        html.P(["Selecciona un tweet del dataset predeterminado para evaluar su polaridad usando aprendizaje máquina."],
               style=text_style_p),
        # Dropdown input
        html.Div([
            html.Label("Selecciona un tweet para ser evaluado"),
            dcc.Dropdown(
                id = 'tweet-dropdown',
                multi = False,
                # Tweet options for prediction
                options=[{
                    'label': tweet,
                    'value': tweet
                } for tweet in dropdown_options],
                value = dropdown_options[7]
            ),
            html.Br(),

            html.Div(id = 'output-dropdown'),
            
            #html.Button("Predecir", id='drop-predict')            
        ], style={'marginTop': 25, 'marginBottom': 25}),
    ], style={'marginLeft': 600, 'marginRight': 600}), # # # TWEET INPUTS # # #

    html.Hr([], style={'marginLeft': 600, 'marginRight': 600}),

    #-#-#-#-#-# WORD2VEC 3D #-#-#-#-#-#
    html.Div([
        html.H4(["Representación vectorial"], style=text_style_h4),
        html.P(["Representación vectorial de las palabras del dataset usando word2vec." +
                "La dimensión de los vectores se ha reducido usando la técnica t-SNE."],
               style=text_style_p),

        html.H5("Word2vec 3D"),

        dcc.Graph(
            figure={
                'data':[
                    go.Scatter3d(
                        x = x_w2v,
                        y = y_w2v,
                        z = z_w2v,
                        text = word_vocab,
                        mode = 'markers',
                        opacity = 0.7,
                        marker = {
                            'size': 5,
                            'line': {'width': 0.5, 'color': 'white'}
                        } # marker
                    ) # go.Scatter
                ] # data
            }, # figure
        ), # graph

        html.Br(),
        html.Br(),

        html.H5("Word2vec 2D"),

        #-#-#-#-#-# WORD2VEC 2D #-#-#-#-#-# 
        dcc.Graph(
            figure={
                'data':[
                    go.Scatter(
                        x = x_w2v,
                        y = y_w2v,
                        text = word_vocab,
                        mode = 'markers',
                        opacity = 0.7,
                        marker = {
                            'size': 5,
                            'line': {'width': 0.5, 'color': 'white'}
                        } # marker
                    ) # go.Scatter
                ] # data
            } # figure
        ), # graph

        html.Br(),
        html.Br()
        
    ], style={'marginLeft': 600, 'marginRight': 600, 'marginBottom': 100}),

    #-#-#-#-#-# FOOTER #-#-#-#-#-#
    html.Div([
    ], style={
        "margin": "10%",
        'backgroundColor': '#F4F4F8 !important'})
]) # layout



# # # # UPDATE VALUES METHODS # # # # #
# # # # # # # # # # # # # # # # # # # #
@app.callback(
    Output('output-dropdown', 'children'),
    [Input('tweet-dropdown', 'value')]
)

def update_output(dropdown):
    # Search value in tweet list
    index = list(test_df["Tweet"].values).index(dropdown)

    # Get prediction value
    prediction = np.argmax(predictions[index])

    if prediction == 0:
        prediction = "Negativo"
    else:
        prediction = "Positivo"

    # Get prediction confidence
    confidence = max(predictions[index])
    confidence = round(confidence, 3)
    
    # Get real polarity value
    polarity = test_df["Polarity"].values[index] // 4

    if polarity == 0:
        polarity = "Negativo"
    else:
        polarity = "Positivo"    

    result = "Predicción: ", prediction, "\t | Confianza: ", confidence, "\t | Polaridad Real: ", polarity

    return result


if __name__ == '__main__':
    app.run_server(debug=True)
