'''
Plots for the sufficient sketching dimension experiments
`subspace_embedding_dimension_.py`
'''
import json
import sys
sys.path.insert(0,'..')
import os
from experiment_parameter_grid import subspace_embedding_exp_setup
from my_plot_styles import plotting_params
from matplotlib_config import update_rcParams
import numpy as np

from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params

update_rcParams()


def plot_distortion_vs_sample(exp_results,n,d, save_dir):


    aspect_ratio = exp_results['aspect ratio']
    num_trials = exp_results['num trials']
    aspect_ratio = str(d/n)

    # Collect all of the distribution keys to iterate over.
    distributions = []
    for d_name in exp_results.keys():
        if d_name != 'aspect ratio' and d_name != 'num trials':
            distributions.append(d_name)

    for dist in distributions:

        sketch_results = exp_results[dist]['experiment']
        coherence_ratio = str(round(exp_results[dist]['coherence'] / exp_results[dist]['min leverage score'],3))
        #print(sketch_results)


        fig, ax = plt.subplots(figsize=(9,6))

        for sketch in sketch_results.keys():
            my_colour = plotting_params[sketch]['colour']
            my_marker = plotting_params[sketch]['marker']
            my_line = plotting_params[sketch]['line_style']
            my_title = dist.capitalize() + ", $\mu = {}, d/n={}$".format(coherence_ratio,aspect_ratio)

            # Extract the plotting data - x --> sampling factors and y --> distortion
            x_vals = np.array(list(sketch_results[sketch].keys())) #sampling factor for x axis
            y_vals = np.array([sketch_results[sketch][key]['mean distortion'] for key in x_vals])

            # Get rank failures
            rank_fail_check = np.array([sketch_results[sketch][key]['rank failures'] for key in x_vals])
            bad_ids = rank_fail_check[rank_fail_check > 0] #np.where(rank_fail_check > 0)
            bad_x = x_vals[rank_fail_check > 0]
            bad_y = y_vals[rank_fail_check > 0]
            good_x = x_vals[rank_fail_check == 0]
            good_y = y_vals[rank_fail_check == 0]
            my_marker_size = [30*i for i in bad_ids]
            ax.scatter(bad_x, bad_y, color=my_colour, marker='x', s=my_marker_size)
            ax.scatter(good_x, good_y, color=my_colour, marker=my_marker)
            ax.plot(x_vals, y_vals, color=my_colour, linestyle=my_line, label=sketch)
            ax.legend(title=my_title,frameon=False, loc="upper right")
        ax.set_xlabel('Sampling factor ($\gamma$)')
        ax.set_ylabel('Distortion ($\epsilon$)')
        #ax.set_ylim(bottom=0)
        #ax.grid()
        #plt.show()
        plot_name = save_dir + "subspace_embedding_dimension_" + dist + "_" + str(n) + "_" + str(d) + ".pdf"
        fig.savefig(plot_name, bbox_inches='tight')
        print("Saved plots for {}".format(dist))



def main():
    directory = "subspace_embedding_results/"
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


    range_to_test = subspace_embedding_exp_setup['aspect ratio range'] #np.concatenate((np.linspace(0.01,0.1,10),np.linspace(0.125, 0.5,4)))
    for n in subspace_embedding_exp_setup['rows']:
        for scale in range_to_test:
            d = np.int(scale*n)
            print("Plotting for n={}, d={}".format(n,d))
            input_file = 'subspace_embedding_dimension_' + str(n) + "_" + str(d) + ".npy"
            results = np.load(input_file)[()]
            plot_distortion_vs_sample(results,n,d,directory)

if __name__ == "__main__":
    main()
