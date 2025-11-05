'''
todo:
- add to full database?
'''

from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.graph_objects as go
import numpy as np
from utils import load_data, main_chemical_embedding, main_taste_embedding, get_taste_vector

df_Custom, chemvecs_mol2vec, chemvecs_morgan = load_data()

options = {
    'chem': {
        'vector_type': 'mol2vec',
        'no_structure_policy': 'remove',
        'no_chemosensory_policy': 'include',
        'z_score': 'False',
        'method': '2D t-SNE',
        'distance_metric': 'cosine',
        'use_full_database': 'False',
        'perplexity': 15
    },
    'taste': {
        'no_chemosensory_policy': 'zero',
        'z_score': 'False',
        'method': '2D t-SNE',
        'distance_metric': 'cosine',
        'use_full_database': 'False',
        'perplexity': 15
    }
}

# highlight mouse diet only
# foods_and_colors = [
#     ('Millet', 'black'),
#     ('Sorghum', 'black'),
#     ('Cucurbita', 'black'),
#     ('Sunflower', 'black'),
#     ('Sesame', 'black'),
#     ('Wheat', 'black'),
#     ('Corn', 'black'),
#     ('Barley', 'black'),
#     ('Rice', 'black'),
#     ('Carrot', 'green'),
#     ('Broccoli', 'green'),
#     ('Spinach', 'green'),
#     ('Chicory', 'green'),
#     ('Garden tomato', 'green'),
#     ('Potato', 'green'),
#     ('Common pea', 'blue'),
#     ('Soy bean', 'blue'),
#     ('Peanut', 'blue'),
#     ('Almond', 'blue'),
#     ('Walnut', 'blue'),
#     ('American cranberry', 'red'),
#     ('Banana', 'red'),
#     ('Black raisin', 'red'),
#     ('Apple', 'red'),
#     ('Red raspberry', 'red'),
#     ('Papaya', 'red'),
#     ('Coconut', 'red'),
#     ('Lowbush blueberry', 'red'),
#     ('Sour cherry', 'red'),
#     ('Sweet cherry', 'red'),
#     ('Grape', 'red'),
#     ('Mango', 'red'),
#     ('Strawberry', 'red'),
#     ('Shrimp', 'cyan'),
#     ('Yogurt', 'yellow'),
#     ('Swiss cheese', 'yellow'),
#     ('Eggs', 'yellow'),
#     ('Breakfast cereal', 'magenta'),
#     ('Chocolate', 'magenta')
# ]

# highlight more
foods_and_colors = [
    ('Millet', 'black'),
    ('Sorghum', 'black'),
    ('Cucurbita', 'green'),
    ('Sunflower', 'black'),
    ('Sesame', 'black'),
    ('Wheat', 'black'),
    ('Corn', 'black'),
    ('Barley', 'black'),
    ('Rice', 'black'),
    ('Oat', 'black'),
    ('Rye', 'black'),
    ('Quinoa', 'black'),
    ('Sorrel', 'black'),
    ('Carrot', 'green'),
    ('Broccoli', 'green'),
    ('Spinach', 'green'),
    ('Chicory', 'green'),
    ('Garden tomato', 'green'),
    ('Potato', 'green'),
    ('Common pea', 'blue'),
    ('Soy bean', 'blue'),
    ('Peanut', 'blue'),
    ('Almond', 'blue'),
    ('Walnut', 'blue'),
    ('Cashew nut', 'blue'),
    ('Pecan nut', 'blue'),
    ('Macadamia nut', 'blue'),
    ('Pineapple', 'red'),
    ('Lemon', 'red'),
    ('Lime', 'red'),
    ('Pear', 'red'),
    ('American cranberry', 'red'),
    ('Banana', 'red'),
    ('Black raisin', 'red'),
    ('Apple', 'red'),
    ('Red raspberry', 'red'),
    ('Papaya', 'red'),
    ('Coconut', 'red'),
    ('Lowbush blueberry', 'red'),
    ('Sour cherry', 'red'),
    ('Sweet cherry', 'red'),
    ('Grape', 'red'),
    ('Mango', 'red'),
    ('Strawberry', 'red'),
    ('Shrimp', 'cyan'),
    ('Yogurt', 'yellow'),
    ('Swiss cheese', 'yellow'),
    ('Parmesan cheese', 'yellow'),
    ('Cheddar Cheese', 'yellow'),
    ('Mozzarella cheese', 'yellow'),
    ('Blue cheese', 'yellow'),
    ('Eggs', 'yellow'),
    ('Breakfast cereal', 'magenta'),
    ('Chocolate', 'magenta')
]

food_names = [tuple_[0] for tuple_ in foods_and_colors]
colors = [tuple_[1] for tuple_ in foods_and_colors]
all_food_names = df_Custom['food_name'].unique()


# main layout (tabs)
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = 'FoodAnalysisApp'

app.layout = html.Div([
    html.H1('Food analysis'),
    dcc.Tabs(id="tabs", value='tab_1', children=[
        dcc.Tab(label='Food visualizations', value='tab_1'),
        dcc.Tab(label='Database', value='tab_2'),
    ], className='navbar'),
    html.Div(id='tabs_content')
])


# tab-specific layouts and tab-switching behavior
@app.callback(
    Output('tabs_content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab_1':

        return html.Div([
            html.H2('Food visualizations'),

            html.H3('Visualization in chemical space'),

            html.Div([
                html.Div('Use full database:'),
                dcc.Dropdown(['True', 'False'], 'False', id='chem_use_full_database'),
                html.Div('Vector type:'),
                dcc.Dropdown(['mol2vec', 'morgan'], 'mol2vec', id='chem_vector_type'),
                html.Div('No structure policy:'),
                dcc.Dropdown(['zero', 'remove'], 'remove', id='chem_no_structure_policy'),
                html.Div('No chemosensory policy:'),
                dcc.Dropdown(['include', 'zero', 'remove'], 'include', id='chem_no_chemosensory_policy'),
                html.Div('z-score:'),
                dcc.Dropdown(['True', 'False'], 'False', id='chem_z_score'),
                html.Div('Method:'),
                dcc.Dropdown(['2D t-SNE', '3D t-SNE', '2D PCA', '3D PCA'], '2D t-SNE', id='chem_method'),
                html.Div('Distance metric:'),
                dcc.Dropdown(['euclidean', 'cosine', 'angle', 'hamming', 'jaccard'], 'cosine', id='chem_distance_metric'),
            ], className='dropdown_container', style={'height': '375px'}),

            html.Div([
                html.Div([
                    html.H3(''),
                    dcc.Graph(id='g1')
                ]),
                html.Div([
                    html.H3(''),
                    dcc.Graph(id='g2')
                ]),
            ], className="row"),

            html.H3('Visualization in taste space'),

            html.Div([
                html.Div('Use full database:'),
                dcc.Dropdown(['True', 'False'], 'False', id='taste_use_full_database'),
                html.Div('No chemosensory policy:'),
                dcc.Dropdown(['zero', 'remove'], 'zero', id='taste_no_chemosensory_policy'),
                html.Div('z-score:'),
                dcc.Dropdown(['True', 'False'], 'False', id='taste_z_score'),
                html.Div('Method:'),
                dcc.Dropdown(['2D t-SNE', '3D t-SNE', '2D PCA', '3D PCA'], '2D t-SNE', id='taste_method'),
                html.Div('Distance metric:'),
                dcc.Dropdown(['euclidean', 'cosine', 'angle'], 'cosine', id='taste_distance_metric'),
            ], className='dropdown_container'),

            html.Div([
                html.Div([
                    html.H3(''),
                    dcc.Graph(id='g3')
                ]),

                html.Div([
                    html.H3(''),
                    dcc.Graph(id='g4')
                ]),
            ], className="row"),

            html.H3('Single food taste plot'),

            html.Div([
                html.Div('Select food:'), 
                dcc.Dropdown(all_food_names, all_food_names[0], id='taste_plot_food'),
                html.Div('No chemosensory policy:'),
                dcc.Dropdown(['zero', 'remove'], 'zero', id='taste_plot_no_chemosensory_policy'),
            ], className='dropdown_container'),

            html.Div([
                dcc.Graph(id='g5')
            ])
        ])
    
    elif tab == 'tab_2':

        return html.Div([
            html.H2('Database'),
            html.Div([
                html.Div('Select food:'), 
                dcc.Dropdown(all_food_names, all_food_names[0], id='food_name')
            ], className='dropdown_container'),
            html.H3('Data'),
            dash_table.DataTable(id='table')
        ])


# chemical space plots updating behavior
@app.callback(
    [Output('g1', 'figure'), 
     Output('g2', 'figure')],
    [Input('chem_use_full_database', 'value'),
     Input('chem_vector_type', 'value'),
     Input('chem_no_structure_policy', 'value'),
     Input('chem_no_chemosensory_policy', 'value'),
     Input('chem_z_score', 'value'),
     Input('chem_method', 'value'),
     Input('chem_distance_metric', 'value')]
)
def update_chem_figures(use_full_database, vector_type, no_structure_policy, no_chemosensory_policy, z_score, method, distance_metric):
    options['chem']['use_full_database'] = use_full_database
    options['chem']['vector_type'] = vector_type
    options['chem']['no_structure_policy'] = no_structure_policy
    options['chem']['no_chemosensory_policy'] = no_chemosensory_policy
    options['chem']['z_score'] = z_score == 'True'
    options['chem']['method'] = method
    options['chem']['distance_metric'] = distance_metric
    D, X, inds_to_render = main_chemical_embedding(food_names, df_Custom, chemvecs_mol2vec, chemvecs_morgan, options)
    # FIGURE 1
    if 't-SNE' in options['chem']['method']:
        mask = np.array([i in inds_to_render for i in range(D.shape[0])])
        mask = mask.reshape([-1, 1]) & mask.reshape([1, -1])
        D = D[mask].reshape([len(inds_to_render), len(inds_to_render)])
        x_labels = food_names
        y_labels = food_names
        hover_template = 'd(%{x}, %{y}) = %{z}<extra></extra>'
        title_text = 'Distance'
        x_axis_text = 'Food'
        y_axis_text = 'Food'
    else:
        x_labels = 1 + np.arange(D.shape[0])
        y_labels = 1 + np.arange(D.shape[0])
        hover_template = 'cov(feature_%{x}, feature_%{y}) = %{z}<extra></extra>'
        title_text = 'Covariance'
        x_axis_text = 'Feature'
        y_axis_text = 'Feature'
    fig1 = go.Figure(data=go.Heatmap(z=D, x=x_labels, y=y_labels))
    fig1.update_traces(hovertemplate=hover_template)
    fig1.update_layout(
        margin=dict(l=50, r=50, b=50, t=50), 
        xaxis_showticklabels=False, 
        yaxis_showticklabels=False,
        title=title_text,
        xaxis_title=x_axis_text,
        yaxis_title=y_axis_text,
        yaxis_autorange='reversed',
        width=500
    )
    # FIGURE 2
    if '2D' in options['chem']['method']:
        fig2 = go.Figure(data=[
            go.Scatter(
                x=X[:, 0], 
                y=X[:, 1], 
                mode='markers', 
                marker=dict(color='grey', size=7),
                hoverinfo='skip',
                showlegend=False
            ),
            go.Scatter(
                x=X[np.array(inds_to_render), 0], 
                y=X[np.array(inds_to_render), 1], 
                mode='markers', 
                marker=dict(color=colors, size=7),
                hovertemplate='%{text}<extra></extra>',
                text=food_names,
                showlegend=False
            )])
        fig2.update_layout(
            margin=dict(l=50, r=50, b=50, t=50),
            title=('Embedding' if 't-SNE' in options['chem']['method'] else 'Projection'),
            xaxis_title=('t-SNE1' if 't-SNE' in options['chem']['method'] else 'PC1'),
            yaxis_title=('t-SNE2' if 't-SNE' in options['chem']['method'] else 'PC2'))
    elif '3D' in options['chem']['method']:
        fig2 = go.Figure(data=[
            go.Scatter3d(
                x=X[:, 0], 
                y=X[:, 1], 
                z=X[:, 2],
                mode='markers', 
                marker=dict(color='grey', size=5),
                hoverinfo='skip',
                showlegend=False
            ),
            go.Scatter3d(
                x=X[np.array(inds_to_render), 0], 
                y=X[np.array(inds_to_render), 1], 
                z=X[np.array(inds_to_render), 2],
                mode='markers', 
                marker=dict(color=colors, size=5),
                hovertemplate='%{text}<extra></extra>',
                text=food_names,
                showlegend=False
            )])
        fig2.update_layout(
            margin=dict(l=50, r=50, b=50, t=50),
            title=('Embedding' if 't-SNE' in options['chem']['method'] else 'Projection'),
            scene=dict(
                xaxis_title=('t-SNE1' if 't-SNE' in options['chem']['method'] else 'PC1'),
                yaxis_title=('t-SNE2' if 't-SNE' in options['chem']['method'] else 'PC2'),
                zaxis_title=('t-SNE3' if 't-SNE' in options['chem']['method'] else 'PC3')
            )
        )
    fig2.update_layout(width=500)

    return fig1, fig2


# taste space plots updating behavior
@app.callback(
    [Output('g3', 'figure'), 
     Output('g4', 'figure')],
    [Input('taste_use_full_database', 'value'),
     Input('taste_no_chemosensory_policy', 'value'),
     Input('taste_z_score', 'value'),
     Input('taste_method', 'value'),
     Input('taste_distance_metric', 'value')]
)
def update_taste_figures(use_full_database, no_chemosensory_policy, z_score, method, distance_metric):
    options['taste']['use_full_database'] = use_full_database
    options['taste']['no_chemosensory_policy'] = no_chemosensory_policy
    options['taste']['z_score'] = z_score == 'True'
    options['taste']['method'] = method
    options['taste']['distance_metric'] = distance_metric
    D, X, inds_to_render = main_taste_embedding(food_names, df_Custom, options)
    # FIGURE 3
    if 't-SNE' in options['taste']['method']:
        mask = np.array([i in inds_to_render for i in range(D.shape[0])])
        mask = mask.reshape([-1, 1]) & mask.reshape([1, -1])
        D = D[mask].reshape([len(inds_to_render), len(inds_to_render)])
        x_labels = food_names
        y_labels = food_names
        hover_template = 'd(%{x}, %{y}) = %{z}<extra></extra>'
        title_text = 'Distance'
        x_axis_text = 'Food'
        y_axis_text = 'Food'
    else:
        x_labels = 1 + np.arange(D.shape[0])
        y_labels = 1 + np.arange(D.shape[0])
        hover_template = 'cov(feature_%{x}, feature_%{y}) = %{z}<extra></extra>'
        title_text = 'Covariance'
        x_axis_text = 'Feature'
        y_axis_text = 'Feature'
    fig3 = go.Figure(data=go.Heatmap(z=D, x=x_labels, y=y_labels))
    fig3.update_traces(hovertemplate=hover_template)
    fig3.update_layout(
        margin=dict(l=50, r=50, b=50, t=50), 
        xaxis_showticklabels=False, 
        yaxis_showticklabels=False,
        title=title_text,
        xaxis_title=x_axis_text,
        yaxis_title=y_axis_text,
        yaxis_autorange='reversed',
        width=500)
    # FIGURE 4
    if '2D' in options['taste']['method']:
        fig4 = go.Figure(data=[
            go.Scatter(
                x=X[:, 0], 
                y=X[:, 1], 
                mode='markers', 
                marker=dict(color='grey', size=7),
                hoverinfo='skip',
                showlegend=False
            ),
            go.Scatter(
                x=X[np.array(inds_to_render), 0], 
                y=X[np.array(inds_to_render), 1], 
                mode='markers', 
                marker=dict(color=colors, size=7),
                hovertemplate='%{text}<extra></extra>',
                text=food_names,
                showlegend=False
            )])
        fig4.update_layout(
            margin=dict(l=50, r=50, b=50, t=50),
            title=('Embedding' if 't-SNE' in options['taste']['method'] else 'Projection'),
            xaxis_title=('t-SNE1' if 't-SNE' in options['taste']['method'] else 'PC1'),
            yaxis_title=('t-SNE2' if 't-SNE' in options['taste']['method'] else 'PC2'))
    elif '3D' in options['taste']['method']:
        fig4 = go.Figure(data=[
            go.Scatter3d(
                x=X[:, 0], 
                y=X[:, 1], 
                z=X[:, 2],
                mode='markers', 
                marker=dict(color='grey', size=5),
                hoverinfo='skip',
                showlegend=False
            ),
            go.Scatter3d(
                x=X[np.array(inds_to_render), 0], 
                y=X[np.array(inds_to_render), 1], 
                z=X[np.array(inds_to_render), 2],
                mode='markers', 
                marker=dict(color=colors, size=5),
                hovertemplate='%{text}<extra></extra>',
                text=food_names,
                showlegend=False
            )])
        fig4.update_layout(
            margin=dict(l=50, r=50, b=50, t=50),
            title=('Embedding' if 't-SNE' in options['taste']['method'] else 'Projection'),
            scene=dict(
                xaxis_title=('t-SNE1' if 't-SNE' in options['taste']['method'] else 'PC1'),
                yaxis_title=('t-SNE2' if 't-SNE' in options['taste']['method'] else 'PC2'),
                zaxis_title=('t-SNE3' if 't-SNE' in options['taste']['method'] else 'PC3')
            )
        )
    fig4.update_layout(width=500)

    return fig3, fig4


# single food taste plot updating behavior
@app.callback(
    Output('g5', 'figure'),
    [Input('taste_plot_food', 'value'),
     Input('taste_plot_no_chemosensory_policy', 'value')]
)
def update_taste_plot(food_name, no_chemosensory_policy):
    u, labels = get_taste_vector(food_name, df_Custom, no_chemosensory_policy)
    # FIGURE 5
    fig5 = go.Figure(data=go.Bar(x=labels, y=u))
    fig5.update_layout(
        margin=dict(l=50, r=50, b=50, t=50), 
        title=food_name,
        yaxis_range=[0, 1],
        width=700)

    return fig5


# database rendering behavior
@app.callback(
    Output('table', 'data'),
    Input('food_name', 'value')
)
def update_table(food_name):
    data = df_Custom[df_Custom['food_name'] == food_name].to_dict('records')
    return data


if __name__ == '__main__':
    app.run(debug=True)