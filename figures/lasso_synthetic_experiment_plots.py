'''
Plots for the LASSO experients `lasso_synthetic_experiment.py`
'''
import json
import sys
sys.path.insert(0,'..')
from experiment_parameter_grid import param_grid
from my_plot_styles import plotting_params
import numpy as np
import matplotlib.pyplot as plt

def plot_lasso_synthetic_d():
    results = np.load('lasso_synthetic_times_d.npy')
    results = results[()]
    fig,ax = plt.subplots(dpi=250)
    for method in results.keys():
        cols = [d for d in results[method].keys()]

        if method == "sklearn":
            times = [results[method][d]["solve time"] for d in results[method].keys()]
            my_label = 'sklearn'
            my_colour = plotting_params['Exact']['colour']
            my_marker = plotting_params['Exact']['marker']
            ax.plot(cols, times, color=my_colour, marker=my_marker,
                            label=my_label, markersize=6.0)

        else:
            times = [results[method][d]["total time "] for d in results[method].keys()]
            my_label = method
            my_colour = plotting_params[method]['colour']
            my_marker = plotting_params[method]['marker']
            ax.plot(cols, times, color=my_colour, marker=my_marker,
                                label=my_label, markersize=6.0)


    ax.legend()
    ax.set_xlabel("d")
    ax.set_yscale('log')
    ax.set_ylabel("log(time) (log seconds)")
    fig.savefig('time_vs_cols.pdf', bbox_inches="tight")
    plt.show()


        #print(times)
        # if method is "sklearn":
        #     times = [t for results[method][d]["solve time"] for d in results[method].keys()]
        #     #ax.plot(cols,)


def main():
    plot_lasso_synthetic_d()

if __name__ == "__main__":
    main()
