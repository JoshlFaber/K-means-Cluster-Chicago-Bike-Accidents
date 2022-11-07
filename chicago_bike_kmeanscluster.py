#%%


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=False)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
    
)
# path_file is the file path for the chicago accident report CSV file 
# N_cluster is total number of clusters we make 
# the density is the lower bound for the number of accidents per cluster that we see in the output map 


def produce_cluster_map(path_file,N_cluster =150,Density = 100):
    df = pd.read_csv(path_file) # path file 
    df = df[df['FIRST_CRASH_TYPE'] == 'PEDALCYCLIST']


    df = df[df['LONGITUDE'] != 0] # we get rid of false values for longitutde and latitude 
    df = df.dropna(subset = ["LATITUDE", "LONGITUDE"]) #we drop rows with NaN in longitutde and latitude 
    X = df.loc[:, ["LATITUDE", "LONGITUDE"]]
    kmeans = KMeans(n_clusters=N_cluster) # this is the number of clusters
    X["Cluster"] = kmeans.fit_predict(X)
    X["Cluster"] = X["Cluster"].astype("category")
    value_dict = {} #creates a count of number of accidents per cluster
    for clst in X["Cluster"]:
        if str(clst) not in value_dict:
            value_dict[str(clst)] = 1
        else:
            value_dict[str(clst)] += 1

    X["Cluster_Count"] = X["LONGITUDE"] # holds a dummy variable to be changed in the next for loop
    for index,row in X.iterrows(): #creates a column that counts the amount of biek accidents in each cluster

        X.at[index,"Cluster_Count"] = int(value_dict[str(int(row["Cluster"]))]) 

        
    X_max = X[X["Cluster_Count"] > Density] # creates a lower bound for how many bike accidents that we see
    

    sns.relplot(
        x="LONGITUDE", y="LATITUDE", hue="Cluster", data=X_max, height=10,
    )

#%%

