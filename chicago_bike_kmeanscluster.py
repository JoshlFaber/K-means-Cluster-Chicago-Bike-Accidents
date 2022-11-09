#%%


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import folium 
from folium import Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd

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


def produce_cluster(path_file,N_cluster =150,Density = 100):
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

    return X_max

def graph_cluster(cluster_data):
    sns.relplot(
        x="LONGITUDE", y="LATITUDE", hue="Cluster", data=cluster_data, height=10,
    )


path = r"C:\Users\joshl\OneDrive\Desktop\TrafficCrashes.csv" ## change this to the file on your computer 
bike_path = r"C:\Users\joshl\OneDrive\Desktop\CDOT_Bike_Routes_2014_1216.csv"
full_bike_path = pd.read_csv(bike_path)
divvy_chicago = r"C:\Users\joshl\OneDrive\Desktop\Divvy_Bicycle_Stations.csv"
divvy_chicago_data = pd.read_csv(divvy_chicago)


## below produces a dictonary of the latitude a longitude of each cluster 
def long_lat_cluster_dict(cluster_data):

    long_lat_dict = {} # key = cluster number, value = list of [[lat_1,long_1],...[lat_n,long_n]]
    for index,row in cluster_data.iterrows():
        if str(row['Cluster']) not in long_lat_dict:
            long_lat_dict[str(row['Cluster'])] = [[row["LATITUDE"],row["LONGITUDE"]]]
        else:
            long_lat_dict[str(row['Cluster'])].append([row["LATITUDE"],row["LONGITUDE"]])
    return(long_lat_dict)

def min_max_cluster_dict(cluster_data):
    min_max_dict = {} #value cluster number, key [[min lat,max lat],[min long, max long]]
    long_lat_D = long_lat_cluster_dict(cluster_data)
    for value in long_lat_D:
        df_long_lat = pd.DataFrame(long_lat_D[value],columns = ['LATITUDE', 'LONGITUDE'])
        min_max_dict[value] = [[df_long_lat['LATITUDE'].min(),df_long_lat['LATITUDE'].max()],[df_long_lat['LONGITUDE'].min(),df_long_lat['LONGITUDE'].max()]]
    return min_max_dict

#below is a marker cluster map of the data 
def marker_cluster_map(file_path, zoom_start_n = 13):
    m_mc = folium.Map(location = [41.92, -87.66],tiles = 'openstreetmap', zoom_start = zoom_start_n)

    mc = MarkerCluster()
    df = pd.read_csv(file_path) # path file 
    df = df[df['FIRST_CRASH_TYPE'] == 'PEDALCYCLIST']


    df = df[df['LONGITUDE'] != 0] # we get rid of false values for longitutde and latitude 
    df = df.dropna(subset = ["LATITUDE", "LONGITUDE"]) #we drop rows with NaN in longitutde and latitude 
    X = df.loc[:, ["LATITUDE", "LONGITUDE"]]
    for idx, row in X.iterrows():
        mc.add_child(Marker([row['LATITUDE'],row['LONGITUDE']]))
    m_mc.add_child(mc)
    #change file path below and unhash to save 
    #m_mc.save(r"C:\Users\joshl\OneDrive\Desktop\TrafficCrashes.html")
    return m_mc

# below is a heat map of the original data    
def heat_map(file_path, zoom_start_n =14, r =14):
    m_hm = folium.Map(location = [41.92, -87.66],tiles = 'openstreetmap', zoom_start = zoom_start_n)
    df = pd.read_csv(file_path) # path file 
    df = df[df['FIRST_CRASH_TYPE'] == 'PEDALCYCLIST']


    df = df[df['LONGITUDE'] != 0] # we get rid of false values for longitutde and latitude 
    df = df.dropna(subset = ["LATITUDE", "LONGITUDE"]) #we drop rows with NaN in longitutde and latitude 
    X = df.loc[:, ["LATITUDE", "LONGITUDE"]]
    HeatMap(data = X[["LATITUDE", "LONGITUDE"]], radius = r).add_to(m_hm)
    
    #change file path below and unhash to save 
    #m_hm.save(r"C:\Users\joshl\OneDrive\Desktop\TrafficCrashes.html")
    return m_hm
def coordinates(string):

    c_string = string.split('((')[1].split('))')[0].split(', ')
    for i in range(len(c_string)):
        c_string[i] = c_string[i].split(' ')
        c_string[i][0] = c_string[i][0].replace('\'','')
        c_string[i][1] = c_string[i][1].replace('\'','')
        c_string[i][0] = c_string[i][0].replace('(','')
        c_string[i][1] = c_string[i][1].replace('(','')
        c_string[i][0] = c_string[i][0].replace(')','')
        c_string[i][1] = c_string[i][1].replace(')','')
        c_string[i][0], c_string[i][1] = float(c_string[i][1]), float(c_string[i][0])
    #print(string)
    return(c_string)

def get_color(bike_lane_type):
    if bike_lane_type == 'BIKE LANE':
        return 'red'
    elif bike_lane_type == 'SHARED-LANE':
        return 'peru'
    elif bike_lane_type == 'BUFFERED BIKE LANE':
        return 'purple'
    elif bike_lane_type == 'PROTECTED BIKE LANE':
        return 'blue'
    elif bike_lane_type == 'NEIGHBORHOOD GREENWAY':
        return 'green'
 
def heat_map_with_bike_paths_and_inservice_divvy_stations(bike_accidents_path = path, bike_file_path = full_bike_path, divvy_path = divvy_chicago_data):
    my_map = folium.Map(location = [41.88451394892348, -87.68977117908742],tiles = 'openstreetmap', zoom_start =14)
    for index,row in bike_file_path.iterrows():
        
        bike_coords = coordinates(bike_file_path['the_geom'].loc[index])
        
        folium.PolyLine(bike_coords,color = get_color(row['DISPLAYROU'])).add_to(my_map) #creates the bike lane lines
    df = pd.read_csv(bike_accidents_path) # path file 
    df = df[df['FIRST_CRASH_TYPE'] == 'PEDALCYCLIST']


    df = df[df['LONGITUDE'] != 0] # we get rid of false values for longitutde and latitude 
    df = df.dropna(subset = ["LATITUDE", "LONGITUDE"]) #we drop rows with NaN in longitutde and latitude 
    X = df.loc[:, ["LATITUDE", "LONGITUDE"]]
    HeatMap(data = X[["LATITUDE", "LONGITUDE"]], radius = 14).add_to(my_map) 
    divvy_in_service = divvy_path[divvy_path['Status'] == 'In Service'] # We add the divvy station data
    for index,row in divvy_in_service.iterrows():
        latitude = divvy_in_service['Latitude'].loc[index]
        longitude = divvy_in_service['Longitude'].loc[index]
        folium.Circle(radius = 2, location = [latitude,longitude]).add_to(my_map)
    # unhash below to save 
    #my_map.save(r"C:\Users\joshl\OneDrive\Desktop\heatmapdivvy.html")
    return my_map



#%%

