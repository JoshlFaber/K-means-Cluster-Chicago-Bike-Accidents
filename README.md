# K-means-Cluster-Chicago-Bike-Accidents
A basic K-Means CLustering algorithm for understanding bicycle accidents in Chicago

In this repo we bring together data sets avaliable in the Chicago Data Portal to see visualize geographic locations against reported bicycle accidents. 

The data sets are,

- The Chicago accident reports focused on the columns that occured with bicycles.
- All the city bike lanes (which we color code by type and viualized using PolyLine from the Folium Python package)
- All city bike racks (represented by pink circle markers from the Folium Python package)
- All in service Divvy Stations (represented by blue circle markers from the Folium Python package)
- All businesses with active bussiness licenses (represented by green circle markers from the Folium Python package)

We represent the bicycle accidents in two ways. One we use a K-means cluster algorithm with a high number of clusters to find highdensity areas, and
the write code to filter out the low denisty areas. 

We make use of the Heat map class from Folium to represent the bicycle accidents in Chicago. The other data sets are then superimposed on the heat map.

We provide different functions for variations of the heatmap (with the bike racks but without the businesses and divvy stations for instance) 

We plan to add to this repo in the future and should be seen as a growing/evolving work. 
