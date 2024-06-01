import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import time

class Visualizer:

    def __init__(self, visualize_vars, summary_filename):
        self.visualize_vars = visualize_vars
        self.var_dict = {var: [] for var in visualize_vars}
        self.app = dash.Dash(__name__)
        self.app.layout = html.Div([
            dcc.Tabs(id="tabs", value=visualize_vars[0], children=[
                dcc.Tab(label=var, value=var) for var in visualize_vars
            ]),
            dcc.Graph(id='graph'),
            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # in milliseconds
                n_intervals=0
            )
        ])
        self.vis_time = 0.0  # Initialize vis_time
        self.wait_time = 0.0

    def update_plot(self, out_dict):
        v_start = time.time()  # Start timing
        for var in self.visualize_vars:
            if var in ['objective', 'optimality', 'feasibility']:
                self.var_dict[var].append(out_dict[var])
            elif var.startswith('jacobian['):
                idx1, idx2 = map(int, var[9:-1].split(','))
                self.var_dict[var].append(out_dict[var.split('[')[0]][idx1, idx2])
            else:
                idx = int(var[var.index('[')+1:-1])
                self.var_dict[var].append(out_dict[var.split('[')[0]][idx])
        
        self.vis_time += time.time() - v_start  # Update vis_time

    def keep_plot(self):
        w_start = time.time()

        @self.app.callback(Output('graph', 'figure'), [Input('interval-component', 'n_intervals')])
        def update_figure(n):
            selected_tab = self.app.layout.tabs.value
            return go.Figure(data=[go.Scatter(x=list(range(len(self.var_dict[selected_tab]))), y=self.var_dict[selected_tab])])

        self.app.run_server(debug=True)

        self.wait_time += time.time() - w_start