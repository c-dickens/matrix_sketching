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
    results = np.load('lasso_synthetic_times_vary_d_at_n_500000.npy')
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
            times = [results[method][d]["total time"] for d in results[method].keys()]
            my_label = method
            my_colour = plotting_params[method]['colour']
            my_marker = plotting_params[method]['marker']
            ax.plot(cols, times, color=my_colour, marker=my_marker,
                                label=my_label, markersize=6.0)

    ax.legend(title="n=5,m=2") # for later usae  ax.legend(title='(n,m) = ({},{}d)'.format(n_rows,sketch_size))
    ax.set_xlabel("d")
    ax.set_yscale('log')
    ax.set_ylabel("log(time) (log seconds)")
    fig.savefig('lasso_time_vs_cols.pdf', bbox_inches="tight")
    plt.show()

def plot_sketch_time_opt_time_vs_d():
    results = np.load('lasso_synthetic_times_vary_d_at_n_500000.npy')
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
            times = [results[method][d]["total time"] for d in results[method].keys()]
            my_label = "Total time: " + method
            my_colour = plotting_params[method]['colour']
            my_marker = plotting_params[method]['marker']
            ax.plot(cols, times, color=my_colour, marker=my_marker,
                                label=my_label, markersize=6.0)

            sketch_times = [results[method][d]["sketch_time"] for d in results[method].keys()]
            sketch_label =  "Sketch time: " + method
            ax.plot(cols, sketch_times, color=my_colour, marker=my_marker,
                    linestyle=":", label=sketch_label, markersize=6.0)

            opt_times = [results[method][d]["optimisation time"] for d in results[method].keys()]
            opt_label =  "Optimisation time: " + method
            ax.plot(cols, opt_times, color=my_colour, marker=my_marker,
                    linestyle="--", label=opt_label, markersize=6.0)


    ax.legend(title="n=5,m=2") # for later usae  ax.legend(title='(n,m) = ({},{}d)'.format(n_rows,sketch_size))
    ax.set_xlabel("d")
    ax.set_yscale('log')
    ax.set_ylabel("log(time) (log seconds)")
    fig.savefig('lasso_time_vs_cols_detailed.pdf', bbox_inches="tight")
    plt.show()



def plot_lasso_synthetic_n():
    results = np.load('lasso_synthetic_times_vary_n_at_d_200.npy')
    results = results[()]
    fig,ax = plt.subplots(dpi=250)
    for method in results.keys():
        rows = [n for n in results[method].keys()]

        if method == "sklearn":
            times = [results[method][n]["solve time"] for n in results[method].keys()]
            my_label = 'sklearn'
            my_colour = plotting_params['Exact']['colour']
            my_marker = plotting_params['Exact']['marker']
            ax.plot(rows, times, color=my_colour, marker=my_marker,
                            label=my_label, markersize=6.0)

        else:
            times = [results[method][n]["total time"] for n in results[method].keys()]
            my_label = method
            my_colour = plotting_params[method]['colour']
            my_marker = plotting_params[method]['marker']
            ax.plot(rows, times, color=my_colour, marker=my_marker,
                                label=my_label, markersize=6.0)


    ax.legend(title="d=200,m=2", loc=2) # for later usae  ax.legend(title='(n,m) = ({},{}d)'.format(n_rows,sketch_size))
    ax.set_xlabel("n")
    ax.set_yscale('log')
    ax.set_ylabel("log(time) (log seconds)")
    fig.savefig('lasso_time_vs_rows.pdf', bbox_inches="tight")
    plt.show()

def plot_sketch_time_opt_time_vs_n():
    results = np.load('lasso_synthetic_times_vary_n_at_d_200.npy')
    results = results[()]
    fig,ax = plt.subplots(dpi=250)
    for method in results.keys():
        rows = [n for n in results[method].keys()]

        if method == "sklearn":
            times = [results[method][n]["solve time"] for n in results[method].keys()]
            my_label = 'sklearn'
            my_colour = plotting_params['Exact']['colour']
            my_marker = plotting_params['Exact']['marker']
            ax.plot(rows, times, color=my_colour, marker=my_marker,
                            label=my_label, markersize=6.0)

        else:
            times = [results[method][n]["total time"] for n in results[method].keys()]
            my_label = "Total time: " + method
            my_colour = plotting_params[method]['colour']
            my_marker = plotting_params[method]['marker']
            ax.plot(rows, times, color=my_colour, marker=my_marker,
                                label=my_label, markersize=6.0)

            sketch_times = [results[method][d]["sketch_time"] for d in results[method].keys()]
            sketch_label =  "Sketch time: " + method
            ax.plot(rows, sketch_times, color=my_colour, marker=my_marker,
                    linestyle=":", label=sketch_label, markersize=6.0)

            opt_times = [results[method][d]["optimisation time"] for d in results[method].keys()]
            opt_label =  "Optimisation time: " + method
            ax.plot(rows, opt_times, color=my_colour, marker=my_marker,
                    linestyle="--", label=opt_label, markersize=6.0)

    ax.legend(title="d=200,m=2", loc=2) # for later usae  ax.legend(title='(n,m) = ({},{}d)'.format(n_rows,sketch_size))
    ax.set_xlabel("n")
    ax.set_yscale('log')
    ax.set_ylabel("log(time) (log seconds)")
    fig.savefig('lasso_time_vs_rows_detailed.pdf', bbox_inches="tight")
    plt.show()


def main():
    plot_sketch_time_opt_time_vs_n()
    #plot_sketch_time_opt_time_vs_d()
    #plot_lasso_synthetic_d()
    #plot_lasso_synthetic_n()

if __name__ == "__main__":
    main()
