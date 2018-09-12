'''
Plots for the sufficient sketching dimension experiments
`subspace_embedding_dimension_real_data.py`
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


def plot_distortion_vs_sample(data_name, exp_results,save_dir):
    #print(exp_results['experiment'])
    #print(exp_results.keys())
    sketches = exp_results['experiment'].keys()
    #print(sketches)
    #print(exp_results['experiment']['CountSketch'])

    fig, ax = plt.subplots(figsize=(9,6))
    for sketch in sketches:
        my_colour = plotting_params[sketch]['colour']
        my_marker = plotting_params[sketch]['marker']
        my_line = plotting_params[sketch]['line_style']
        my_title = data_name#.capitalize()

        # Extract the plotting data - x --> sampling factors and y --> distortion
        x_vals = np.array(list(exp_results['experiment'][sketch].keys())) #sampling factor for x axis
        y_vals = np.array([exp_results['experiment'][sketch][key]['mean distortion'] for key in x_vals])
        #print(x_vals)
        #print(y_vals)
        # Get rank failures
        rank_fail_check = np.array([exp_results['experiment'][sketch][key]['rank failures'] for key in x_vals])
        bad_ids = rank_fail_check[rank_fail_check > 0] #np.where(rank_fail_check > 0)
        bad_x = x_vals[rank_fail_check > 0]
        bad_y = y_vals[rank_fail_check > 0]
        good_x = x_vals[rank_fail_check == 0]
        good_y = y_vals[rank_fail_check == 0]
        my_marker_size = [30*i for i in bad_ids]
        ax.scatter(bad_x, bad_y, color=my_colour, marker='x', s=my_marker_size)
        ax.scatter(good_x, good_y, color=my_colour, marker=my_marker)
        ax.plot(x_vals, y_vals, color=my_colour, linestyle=my_line, label=sketch)
        ax.legend(title=my_title,frameon=False, loc="upper left")
    ax.set_xlabel('Sampling factor ($\gamma$)')
    ax.set_ylabel('Distortion ($\epsilon$)')
    #ax.set_ylim(bottom=0)
    #ax.grid()
    plt.show()
    plot_name = save_dir + "subspace_embedding_dimension_" + data_name + ".pdf"
    fig.savefig(plot_name, bbox_inches='tight')
    print("Saved plots for {}".format(data_name))



def main():
    directory = "subspace_embedding_results/"
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)



    input_file = 'subspace_embedding_dimension_real_data.npy'
    results = np.load(input_file)[()]
    for data in results.keys():
        print(data)
        if len(results[data]) != 0:
            print("Plotting for {}".format(data))
            if data == "california_housing_train":
                data_name = "california housing"
                print("Changed name for cali")
            else:
                data_name = data
            plot_distortion_vs_sample(data_name, results[data], directory)

if __name__ == "__main__":
    main()
