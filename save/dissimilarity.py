import sklearn
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import cv2

from numpy.linalg import inv

from scipy.spatial.distance import euclidean,cityblock,squareform
from scipy.cluster.hierarchy import dendrogram, linkage, ward

from fastdtw import fastdtw

from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation, SpectralClustering
import scipy.spatial.distance as ssd

def discretize(square_side_size,trajectory):
    Discretized_trajectory = []
    for point in trajectory:
        i = int(point[1]/square_side_size)
        j = int(point[0]/square_side_size)
        square = [i,j]
        if  len(Discretized_trajectory) == 0 or Discretized_trajectory[-1] != square:
            Discretized_trajectory.append(square)
    return Discretized_trajectory

def distance_matrix(trajectories):
    l =len(trajectories)
    D = np.zeros((l,l))
    for i in range(l):
        for j in range(i,l):
            #D[i,j] = my_affinity_wrapper(trajectories[i],trajectories[j])
            #D[i,j],p = fastdtw(trajectories[i],trajectories[j], dist=euclidean)
            D[i,j],p = fastdtw(trajectories[i],trajectories[j], dist=cityblock)
            D[j,i] = D[i,j]
    return D
	
data_file = "laurier.sqlite"
data_base = sqlite3.connect(data_file)
cursor = data_base.cursor()
#data_base.commit()

db_request1 = "SELECT max(trajectory_id), min(trajectory_id) FROM positions"
max_min = cursor.execute(db_request1).fetchall()
nb_trajectory = max_min[0][0]- max_min[0][1] +1

trajectories = np.array([
    cursor.execute(
        "SELECT p.x_coordinate,p.y_coordinate \
        FROM positions as p \
        WHERE trajectory_id ="+str(i)+" ORDER BY p.frame_number ASC"
    ).fetchall() for i in range(nb_trajectory)
])

H = np.loadtxt("laurier-homography.txt")
h = inv(H)

pixel_trajectories = []
for trajectory in trajectories:
    new_points = []
    for point in trajectory:
        new_point = np.array([point[0],point[1],1])
        new_point = np.matmul(h,new_point)
        new_point = [new_point[0]/new_point[2],new_point[1]/new_point[2]]
        new_points.append(new_point)
    pixel_trajectories.append(new_points)
	
square_side_size = 10
discretized_pixel_trajectories = [discretize(square_side_size,trajectory) for trajectory in pixel_trajectories]


times=[]
start = time.time()
D = distance_matrix(discretized_pixel_trajectories[:])
end = time.time()-start
times.append(end)
print(end)

np.savetxt("Distances/cityblock_10.txt",D)