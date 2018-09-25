'''Plotting script for time_error_ihs.py'''
import numpy as np
from pprint import PrettyPrinter
import itertools
import os
import sys
sys.path.append("..")
from experiment_parameter_grid import ihs_sketches, time_error_ihs_grid
from my_plot_styles import plotting_params, my_lines
from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params



def make_time_plots(n,d,save_dir, time_results):
    file_suffix = "_" + str(n) + "_" + str(d) + ".pdf"
    times2test = time_error_ihs_grid['times']
    sampling_factors = time_error_ihs_grid['sketch_factors']

    # Sklearn constants
    sklearn_time = time_results["Sklearn"]["solve time"]


    # Error to opt plots
    error_file_name = save_dir + "error_time" + file_suffix
    fig, ax = plt.subplots(figsize=(12,9))
    for sketch_method in ihs_sketches:
        my_colour = plotting_params[sketch_method]["colour"]
        for gamma in sampling_factors:
            my_label = sketch_method + str(gamma)
            my_line   = my_lines[sampling_factors.index(gamma)]
            yvals = [np.log10(time_results[sketch_method][gamma][time]["error to opt"]) for time in times2test]
            ax.plot(times2test, yvals, color=my_colour,linestyle=my_line, label=my_label)
    ax.axvline(sklearn_time,color=plotting_params["Exact"]["colour"],label='Sklearn')
    ax.legend(title="({},{})".format(n,d))
    ax.set_ylabel("$\log (\| \hat{x} - x_{OPT} \|^2/n )$")
    ax.set_xlabel('log(Time(seconds))')
    ax.set_xscale('log')
    fig.savefig(error_file_name,bbox_inches="tight")
    plt.show()


    # Number of iterations plots
    num_iters_file_name = save_dir + "num_iters_time" + file_suffix
    fig, ax = plt.subplots(figsize=(12,9))
    for sketch_method in ihs_sketches:
        my_colour = plotting_params[sketch_method]["colour"]
        for gamma in sampling_factors:
            my_label = sketch_method + str(gamma)
            my_line   = my_lines[sampling_factors.index(gamma)]
            yvals = [time_results[sketch_method][gamma][time]["num iterations"] for time in times2test]
            ax.plot(times2test, yvals, color=my_colour, linestyle=my_line, label=my_label)
    ax.legend(title="({},{})".format(n,d))
    ax.set_ylabel("Number of iterations")
    ax.set_xlabel('Time (seconds)')
    #ax.set_xscale('log')
    fig.savefig(num_iters_file_name,bbox_inches="tight")
    plt.show()



def main():
    directory = "time_error_ihs_results/"
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Creating directory. ' +  directory)

    n_trials = time_error_ihs_grid['num trials']
    # for n,d in itertools.product(time_error_ihs_grid['rows'],time_error_ihs_grid['columns']):
    for n,d in itertools.product([100_000],[100,75,50,10]):
        print("Plotting for n,d = {},{}".format(n,d))
        file_name = 'ihs_fixed_time_' + str(n) + '_' + str(d) + '.npy'
        exp_results = np.load(file_name)[()]
        pretty = PrettyPrinter(indent=4)
        pretty.pprint(exp_results)
        make_time_plots(n,d,directory,exp_results)



if __name__ == "__main__":
    main()
