'''
Dict which contains the info to download UCI ML Repo datasets
'target_col' is the location of the target variables in the dataset in python
index notation.
'''

all_datasets = {
      "YearPredictionMSD" : {
                  "url" : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
                  "outputFileName" : '../YearPredictionMSD',
                  'target_col' : 0,
                  'matlab' : False
                 },
       "w1a" : {
                   "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w1a',
                   "outputFileName" : '../w1a',
                   'target_col' : 'libsvm',
                   'matlab' : False
                  },
       "w2a" : {
                   "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w2a',
                   "outputFileName" : '../w2a',
                   'target_col' : 'libsvm',
                   'matlab' : False
                  },
       "w3a" : {
                   "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w3a',
                   "outputFileName" : '../w3a',
                   'target_col' : 'libsvm',
                   'matlab' : False
                  },
       "w4a" : {
                   "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w4a',
                   "outputFileName" : '../w4a',
                   'target_col' : 'libsvm',
                   'matlab' : False
                  },
       "w5a" : {
                   "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w5a',
                   "outputFileName" : '../w5a',
                   'target_col' : 'libsvm',
                   'matlab' : False
                  },
        "w6a" : {
                    "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w6a',
                    "outputFileName" : '../w6a',
                    'target_col' : 'libsvm',
                    'matlab' : False
                   },
        "w7a" : {
                    "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w7a',
                    "outputFileName" : '../w7a',
                    'target_col' : 'libsvm',
                    'matlab' : False
                   },
        "w8a" : {
                    "url" : 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#w8a',
                    "outputFileName" : '../w8a',
                    'target_col' : 'libsvm',
                    'matlab' : False
                   },
       "slice" : {
                    'url' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip',
                    'outputFileName' : '../Slice',
                    'header' : True,
                    'header_location' : 0,
                    'target_col' : -1,
                    'matlab' : False
                  },
       "susy"   :{
                'url' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz',
                'outputFileName' : '../Susy',
                'target_col' : 0,
                'matlab' : False
       },


       "landmark" : {
                  'url'   : 'https://sparse.tamu.edu/Pereyra/landmark',
                  'outputFileName' : '../Landmark',
                  'target_col' : -1,
                  'matlab'  : True,
                  'inputFileName' : 'landmark_data.mat'
       },

       "complex" : {
                  'url'   : 'https://sparse.tamu.edu/JGD_Homology/ch7-7-b2',
                  'outputFileName' : '../Complex',
                  'target_col' : -1,
                  'matlab'  : True,
                  'inputFileName' : 'complex_data.mat'
       },

       "census" : {
                  'url'   : 'https://github.com/chocjy/randomized-quantile-regression-solvers/blob/master/matlab/data/census_data.mat',
                  'outputFileName' : '../Census',
                  'target_col' : -1,
                  'matlab'  : True,
                  'inputFileName' : 'census_data.mat'
       },
       "rail2586" : {
                  'url'   : 'https://sparse.tamu.edu/Mittelmann/rail2586',
                  'outputFileName' : '../rail2586',
                  'target_col' : -1,
                  'matlab'  : True,
                  'inputFileName' : 'rail2586.mat'
        }
}
