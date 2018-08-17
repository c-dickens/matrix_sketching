'''
file containing the plotting information for matplotlib functions.
Includes line styles, colours etc for each of the sketching methods.
'''
from experiment_parameter_grid import param_grid


plotting_params = {"CountSketch" : {"colour" : "b",
                                    "line_style" : '-',
                                    "marker" : "o" },
                   "SRHT" : {"colour" : "k",
                             "marker" : "s",
                             "line_style" : ':'},
                   "Gaussian" : {"colour" : "r",
                                 "marker" : "v",
                                 "line_style" : "-."},
                   "Classical" : {"colour" : "m",
                                  "marker" : "*"},
                    "Exact" : {"colour" : "teal",
                               "marker" : "^"}
                                  }

# nb. the marker styles are for the plots with multiple sketch settings.
my_markers = ['.', 's', '^', 'D', 'x', '+', 'V', 'o', '*', 'H']
col_markers = {param_grid['columns'][i]: my_markers[i] for i in range(len(param_grid['columns']))}
#print(col_markers)
