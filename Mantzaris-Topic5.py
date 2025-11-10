import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from matplotlib import pyplot as plt



# Read data on European coyntries from file europe.csv into a pandas dataframe.
countries = pd.read_csv("europe.csv", sep=",", header=0, quotechar='"')

#################################################################################
#
# Preprocessing
#
#################################################################################



# Set as indexes the country names. This is done so as to display country
# names on various plots later.
countries = countries.set_index('Country')

# Keep only the required features: Unemployment and Inflation
countryData = countries[['Unemployment', 'Inflation']]


#################################################################################
#
# K-means clustering
#
#################################################################################

# Execute k-means using k=3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=20)
clusters = kmeans.fit(countryData)

# Plot data and clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(countryData['Unemployment'], countryData['Inflation'], c=kmeans.labels_.astype(float))

for i, c in enumerate(countryData.index):
    plt.annotate(c, (countryData.loc[c, 'Unemployment'], countryData.loc[c, 'Inflation']), xytext=(9, 9), textcoords='offset points',
                    fontsize=8, arrowprops=dict(arrowstyle="-"))


clusterNames = []    
for i, cntr in enumerate(kmeans.cluster_centers_):
    plt.scatter(cntr[0], cntr[1], marker = "o", s=18, linewidths=5, zorder=10, c="red")
    clusterNames.append('Cluster ' + str(i+1))  

  
plt.legend(handles=scatter.legend_elements()[0], labels=clusterNames)

plt.title('Clusters of european countries')
plt.xlabel('Unemployment')
plt.ylabel('Inflation')
plt.show()


#################################################################################
#
# Hierarchical clustering
#
#################################################################################

# Doing the actual hierarchical agglomerative clustering using the complete linkage.
hcLinkageMatrix = linkage(countryData, "complete", metric="euclidean")

# Show the initial dendrogram
plt.figure(figsize=(10,6))
denrogramFigure = dendrogram(hcLinkageMatrix, leaf_rotation=90, leaf_font_size=8, labels=countries.index)
plt.title('Linkage Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Cluster Distance')
plt.show()

maxDistance = max(denrogramFigure['dcoord'][-1])

# Show a trivial interface to cut the dendrogram at height the
# user specifies. The resulting clusters are also displayed.
while (True):
      try: 
          h = float(input(f'Enter height <={"{:.3f}".format(maxDistance)} to cut tree and generate clusters (negative to quit):'))
          if h < 0:
             print('\nTerminating. ByeBye\n')
             break
            
      except Exception as e:
          print('???')
          continue
      
      clusterLabels = cut_tree(hcLinkageMatrix, height=h).flatten()
      for i in range(max(clusterLabels)+1):
          nms = countries.iloc[list(np.where(clusterLabels==i)[0]), :]
          print(f'\tCluster {i}: ', end='')
          print(', '.join(list(nms.index.astype(str))) )

      plt.figure(figsize=(10,6))
      dendrogram(hcLinkageMatrix, leaf_rotation=90, leaf_font_size=8, labels=countries.index)
      plt.title('Linkage Dendrogram')
      plt.xlabel('Data Points')
      plt.ylabel('Cluster Distance')
      
      # Plot also the height where clusters are generated
      plt.axhline(y=h, color='r', linestyle='--')
      plt.text(0, h, str(h), color='r', va='top', ha='right')
      plt.show()
      
          
