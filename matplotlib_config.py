'''
matplotlib config file for plots.
'''

import matplotlib as mpl

#mpl.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import rc



### Correct format for latex in matplotib from nik
# import matplotlib
# matplotlib.use(‘Agg’)
# import matplotlib.pyplot as plt
# from matplotlib import rc
#
# # Use latex font.
# # rc(‘font’, **{‘family’: ‘serif’, ‘serif’: [‘Computer Modern’]})
# # rc(‘text’, usetex=True)

def update_rcParams():
    # This mpl style is from the UCSC BME163 class.
    plt.rcParams.update({
        'font.size'           : 12.0      ,
        'font.family'         : 'DejaVu Sans'   ,
        'xtick.major.size'    : 4        ,
        'xtick.major.width'   : 0.75     ,
        'xtick.labelsize'     : 12.0      ,
        'xtick.direction'     : 'out'      ,
        'ytick.major.size'    : 4        ,
        'ytick.major.width'   : 0.75     ,
        'ytick.labelsize'     : 12.0      ,
        'ytick.direction'     : 'out'      ,
        'xtick.major.pad'     : 2        ,
        'xtick.minor.pad'     : 2        ,
        'ytick.major.pad'     : 2        ,
        'ytick.minor.pad'     : 2        ,
        'savefig.dpi'         : 900      ,
        'axes.linewidth'      : 0.75     ,
        'text.usetex'         : True     ,
        'text.latex.unicode'  : False     })
