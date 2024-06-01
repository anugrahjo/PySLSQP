import warnings
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import FigureWidget

class Visualizer:

    def __init__(self, visualize_vars, summary_filename):
        '''
        Initialize the visualizer with the variables to visualize. 
        The variables should be a list of strings, where each string is the name of a variable to visualize. 
        The variables can be any of the following:
            - 'objective'       : the objective function value
            - 'optimality'      : the optimality condition
            - 'feasibility'     : the feasibility condition
            - 'x[i]'            : the ith variable value
            - 'constraints[i]'  : the ith constraint value
            - 'jacobian[i,j]'   : the (i,j) element of the Jacobian matrix
            - 'gradient[i]'     : the ith gradient value
            - 'multipliers[i]'  : the ith Lagrange multiplier value

        Creates an interactive plot with the specified variables on the y-axis and the iteration number on the x-axis.
        The plots are stacked vertically in the order they are specified in the list.
        The plot is updated with the latest values of the variables after each iteration.
        '''

        v_start = time.time()
        self.visualize_vars = visualize_vars
        self.fig = make_subplots(rows=len(visualize_vars), cols=1)
        self.fig = FigureWidget(self.fig)  # Convert to FigureWidget to enable real-time updates
        self.fig.update_layout(height=500, width=700, title_text=f'SLSQP Optimization [{summary_filename}]')
        var_dict = {}
        for i, var in enumerate(visualize_vars):
            var_dict[var] = []
            self.fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name=var), row=i+1, col=1)
        self.var_dict = var_dict
        self.vis_time = time.time() - v_start
        self.wait_time = 0.0

    def update_plot(self, out_dict):
        '''
        Update the plot with the latest values of the variables.
        Appends the values of scalar iterates after each iteration in the var_dict attribute before updating the plot.
        The out_dict should be a dictionary containing the following
        keys:
            - 'majiter'     : the number of major iterations
            - 'objective'   : the objective function value
            - 'optimality'  : the optimality condition
            - 'feasibility' : the feasibility condition
            - 'x'           : the variable values
            - 'constraints' : the constraint values
            - 'jacobian'    : the Jacobian matrix
            - 'gradient'    : the gradient values
            - 'multipliers' : the Lagrange multiplier values
        '''

        v_start = time.time()
        x_data = np.arange(out_dict['majiter']+1)
        for k, var in enumerate(self.visualize_vars):
            if var in ['objective', 'optimality', 'feasibility']:
                self.var_dict[var].append(out_dict[var])
            elif var.startswith('jacobian['):
                idx1, idx2 = map(int, var[9:-1].split(','))
                self.var_dict[var].append(out_dict[var.split('[')[0]][idx1, idx2])
            else:
                if var.startswith('x['):
                    idx = int(var[2:-1])
                elif var.startswith('constraints['):
                    idx = int(var[12:-1])
                elif var.startswith('gradient['):
                    idx = int(var[9:-1])
                elif var.startswith('multipliers['):
                    idx = int(var[12:-1])
                self.var_dict[var].append(out_dict[var.split('[')[0]][idx])
            
            # self.fig.update_traces(go.Scatter(x=x_data, y=self.var_dict[var]), selector=dict(name=var))
            self.fig.data[k].x = list(range(len(self.var_dict[var])))
            self.fig.data[k].y = self.var_dict[var]

        time.sleep(1.0)
        self.vis_time += time.time() - v_start

    def keep_plot(self):
        '''
        Keep the plot open after the optimization is completed.
        '''
        w_start = time.time()
        # self.fig.show()
        self.wait_time += time.time() - w_start