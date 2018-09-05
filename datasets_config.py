# Config file to easily extract various datasets
import os.path
from data.get_data.uci_datasets_download_info import all_datasets

num_repeats = 5

datasets =   {  "california_housing_train" : {
        "filepath" : "data/california_housing_train.npy",
        "repeats"  : num_repeats,
    }
}

for data_name in all_datasets.keys():
    file_name = all_datasets[data_name]['outputFileName'][3:]
    datasets[data_name] = {'filepath' : 'data/' + file_name + '.npy',
                           'repeats'  : num_repeats}
    #print('data/' + file_name + '_sparse.npz')
    #print(os.path.isfile('data/' + file_name + '_sparse.npz'))
    if os.path.isfile('data/' + file_name + '_sparse.npz'):
        datasets[data_name]['filepath_sparse'] = 'data/' + file_name + '_sparse.npz'
