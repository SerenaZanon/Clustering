import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import time
from sklearn.cluster import MeanShift
from sklearn.metrics.cluster import rand_score
import json

# 'Mean_model' takes in input the training set and the corresponding labels
# Its goal is to collect information about the executions while changing
    # - the number of Principal Components
    # - the width of the kernel
# It returns a list of dictionaries with all the computed information

def MeanShift_model(X, y):

    # 'data' will be a list of dictionaries containing info about the different executions
    data = []
    
    # Changing the number of adopted features
    for n_features in range(2, 201, 10):
        
        # Creation of a dictionary instance
        elem = {
            'PCA': n_features,
            'outcomes':[],
        }
        
        # Computing Pricipal Component Analysis
        pca = PCA(n_components = n_features)
        print("Computing PCA with ", n_features," components ...")
        X_pca = pca.fit_transform(X)
        
        # Changing the kernel width
        for width in [0.2, 0.4, 0.6, 0.8, 1, 2, 5, 10, 15, 20]:
            
            # Creating the model
            print("Mean Shift with ", width, "of kernel width ...")
            model = MeanShift(bandwidth = width, n_jobs = -1)
            
            # Fit phase
            print("Fitting ...")
            start_fit = time.time()
            model.fit(X_pca)
            end_fit = time.time()
            elapsed_fit = end_fit - start_fit
            
            # Predict phase
            print("Predicting ...")
            start_predict = time.time()
            predictions = model.predict(X_pca)
            end_predict = time.time()
            elapsed_predict = end_predict - start_predict
            
            rand_index = rand_score(y, predictions)
            
            # Filling the dictionary instance
            elem['outcomes'].append({
                'width': width,
                'n_clusters': len(np.unique(predictions)),
                'rand_index': rand_index,
                'fit_time': elapsed_fit,
                'predict_time': elapsed_predict,
            })
    
        data.append(elem)
    
    return data


# Retrieving the DataFrames
X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')

# Printing DataFrames' shape
print("X shape:", X.shape)
print("y shape:", y.shape)

data = MeanShift_model(X, y.to_numpy().ravel())

# Saving json version
with open('./results/data_ms.json', 'w') as file:
    json.dump(data, file, indent=4)
