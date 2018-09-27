import pandas as pd
import numpy as np 


cc = np.loadtxt("colors")
pdf = pd.read_csv("pca", index_col=False )
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, row in pdf.iterrows():
    #plt.scatter(row[0], row[1], color = colors[clusters[i]])

    ax.scatter(row[1], row[2],  row[3], color = cc[i] )
plt.show()