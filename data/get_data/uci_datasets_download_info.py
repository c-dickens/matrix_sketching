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
       "kdd"    :{
                'url' : 'https://www.openml.org/data/get_csv/53996/KDDCup99.arff',
                'outputFileName' : '../KDD',
                'target_col' : -1,
                'categorical columns' : [1,2,3, -1],
                'matlab' : False
       },
       # 'boston'     :{
       #          'url' : 'https://www.openml.org/data/download/52643/boston.arff',
       #          'outputFileName' : '../Boston',
       #          'target_col' : -1,
       #          'matlab' : False
       # },
       # 'mnist'      :{
       #          'url' : 'https://www.openml.org/data/download/18689782/mnist.arff',
       #          'outputFileName' : '../MNIST',
       #          'target_col' : -1,
       #          'matlab' : False
       #
       # },
       "census" : {
                  'url'   : 'https://github.com/chocjy/randomized-quantile-regression-solvers/blob/master/matlab/data/census_data.mat',
                  'outputFileName' : '../Census',
                  'target_col' : -1,
                  'matlab'  : True,
                  'inputFileName' : 'census_data.mat'
       },
       "rucci" : {
                'url' : 'https://sparse.tamu.edu/Rucci/Rucci1',
                'outputFileName' : '../Rucci',
                'target_col'     : -1,
                'matlab'         : True,
                'inputFileName'  : 'Rucci1.mat'
       },
       "rail2586" : {
                  'url'   : 'https://sparse.tamu.edu/Mittelmann/rail2586',
                  'outputFileName' : '../rail2586',
                  'target_col' : -1,
                  'matlab'  : True,
                  'inputFileName' : 'rail2586.mat'
}
}
