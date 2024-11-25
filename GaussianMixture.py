import pandas as pd
from sklearn.decomposition import PCA
import time
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import rand_score
import json

# 'Gaussian_model' takes in input the training set and the corresponding labels
# Its goal is to collect information about the executions while changing
    # - the number of Principal Components
    # - the number of clusters
# It returns a list of dictionaries with all the computed information

def Gaussian_model(X, y):

    # 'data' will be a list of dictionaries containing info about the different executions
    data = []
    
    # Changing the number of adopted features
    for n_features in range(2, 201, 5):
        
        # Creation of a dictionary instance
        elem = {
            'PCA': n_features,
            'outcomes':[],
        }
        
        # Computing Pricipal Component Analysis
        pca = PCA(n_components = n_features)
        print("Computing PCA with ", n_features," components ...")
        X_pca = pca.fit_transform(X)
        
        # Changing the number of clusters
        for n_clusters in range(5, 16):
            
            # Creating the model
            print("Gaussian with ", n_clusters, " clusters ...")
            model = GaussianMixture(n_components = n_clusters, covariance_type = "diag", random_state = 42, max_iter = 300)
            
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
                'n_clusters': n_clusters,
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

data = Gaussian_model(X, y.to_numpy().ravel())

# Saving json version
with open('./results/data_gm.json', 'w') as file:
    json.dump(data, file, indent=4)
