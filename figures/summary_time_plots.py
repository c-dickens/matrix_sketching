'''plotting script for summary_time_experiments.py on synthetic data'''
from pprint import PrettyPrinter
import itertools
import numpy as np
import sys
sys.path.append("..")
from experiment_parameter_grid import param_grid
from matplotlib_config import my_markers
from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params
update_rcParams()




def plot_summary_times(n_rows, data_dict):
    col_list = [10, 100, 1000, 5000]

    fig, ax = plt.subplots(dpi=250)
    for sketch,d in itertools.product(data_dict[n_rows].keys(), col_list):
        print("Plotting: ", n_rows,sketch,d)
        my_colour = plotting_params[sketch]["colour"]
        my_label = str(d)
        my_line = plotting_params[sketch]["line_style"]
        # this just pulls the index of d in the param list and uses that as
        # the marker inded
        my_marker = my_markers[col_list.index(d)]

        ax.plot(data_dict[n_rows][sketch][d].keys(), data_dict[n_rows][sketch][d].values(),
                    color=my_colour, linestyle=my_line, linewidth=2.0,
                    marker=my_marker, markersize=8.0,label=my_label)
    #ax.legend(title='$n$ = {}'.format(n),loc=1)
    ax.set_yscale('log')
    ax.set_ylabel('log(seconds)')
    ax.set_xlabel('Density')
    save_name = "summary_time_density_"+str(n_rows)+".pdf"
    fig.savefig(save_name, bbox_inches="tight")


def main():

    for n_rows in param_grid['rows']:
        print(n_rows)
        #summary_time_vs_sparsity_50000.npy
        file_name = "summary_time_sparsity_"+str(n_rows)+".npy"
        data2plot = np.load('summary_time_vs_sparsity_25000.npy')[()]
        pretty = PrettyPrinter(indent=4)
        pretty.pprint(data2plot)
        plot_summary_times(n_rows, data2plot)

if __name__ == "__main__":
    main()
