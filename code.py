import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# Load the data
data = pd.read_csv('iris-dataset.csv')
# Check the data
data

# create a scatter plot based on two corresponding features (sepal_length and sepal_width; OR petal_length and petal_width)
plt.scatter(data['sepal_length'],data['sepal_width'])
# name your axes
plt.xlabel('Lenght of sepal')
plt.ylabel('Width of sepal')
plt.show()


# create a variable which will contain the data for the clustering
x = data.copy()

# import some preprocessing module
from sklearn import preprocessing

# scale the data for better results
x_scaled = preprocessing.scale(data)
type(data)
#type(x_scaled)

kmeans3c = KMeans(3)
kmeans3c.fit(x_scaled)
clusters3c = x.copy()
clusters3c['cluster_pred'] = kmeans3c.fit_predict(x_scaled)
clusters3c

plt.scatter(clusters3c['sepal_length'],clusters3c['sepal_width'], c=clusters3c['cluster_pred'],cmap='rainbow')
plt.xlabel('Lenght of sepal')
plt.ylabel('Width of sepal')
plt.show()

centroids = kmeans3c.cluster_centers_
centroids

x_scaled_df= pd.DataFrame(data=x_scaled, columns=['sepal_length','sepal_width','a','b'])
x_scaled_df['cluster_pred'] = clusters3c['cluster_pred']
x_scaled_df


plt.scatter(x_scaled_df['sepal_length'],x_scaled_df['sepal_width'], c=x_scaled_df['cluster_pred'],cmap='rainbow')
plt.xlabel('Lenght of sepal')
plt.ylabel('Width of sepal')
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='black', zorder=10)
plt.title('K-means clustering ')

plt.show()
plt.show()