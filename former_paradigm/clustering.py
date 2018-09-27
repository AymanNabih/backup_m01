import sqlite3
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean,cityblock,squareform
from fastdtw import fastdtw
import utils as ut
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns

class Trajectory:
	def __init__(self,trajectory):
		self.meter_trajectory = trajectory
		self.pixel_trajectory = []
		self.discretized_trajectory = []
		self.original_length = len(trajectory)
		self.pixel = False
		self.discretized = 0
		self.discretized_length = 0
		self.extracted_features = []
		self.standardized_features = []

	def to_pixel(self,homography):
		if not self.pixel:
			pixel_points = []
			for point in self.meter_trajectory:
				new_point = np.array([point[0],point[1],1])
				new_point = np.matmul(homography,new_point)
				new_point = [new_point[0]/new_point[2],new_point[1]/new_point[2]]
				pixel_points.append(new_point)
			self.pixel_trajectory = pixel_points
			self.pixel = True
		
	def discretize(self,square_side_size = 50):
		if square_side_size != self.discretized:
			if self.pixel:
				for point in self.pixel_trajectory:
					i = int(point[1]/square_side_size)
					j = int(point[0]/square_side_size)
					square = [i,j]
					if  self.discretized_length == 0 or self.discretized_trajectory[-1] != square:
						self.discretized_trajectory.append(square)
				self.discretized = square_side_size
				self.discretized_length = len(self.discretized_trajectory)
			else:
				print("Error: Trajectory: discretize: convert to pixel first")

	def compute_features(self):
		trajectory_x = [e[0] for e in self.pixel_trajectory] 
		trajectory_y = [e[1] for e in self.pixel_trajectory] 
		velocities = ut.compute_velocities(self.pixel_trajectory) 
		rots = ut.compute_ROTS(self.pixel_trajectory) 
		nb_points = self.original_length 
		distances = euclidean(self.pixel_trajectory[-1],self.pixel_trajectory[0])
		diff_x = ut.get_difference(trajectory_x)
		diff_y = ut.get_difference(trajectory_y)
		diff_v = ut.get_difference(velocities)
		diff_r = ut.get_difference(rots)
		s = [ trajectory_x,trajectory_y,velocities,rots,diff_x,diff_y,diff_v,diff_r ]

		start = time.time()
		features = []
		features.append(nb_points)
		features.append(distances)
			
		for d in s:
			df = pd.DataFrame(d)
			df = df.describe().drop(['count','std']).values.flatten()
			for v in df:
				features.append(v)		
		end = time.time() - start

		self.extracted_features = features



class Clustering:
	def __init__(self,filename,homography):
		self.filename = filename
		self.homography = inv(np.loadtxt(homography))
		self.trajectories = []
		self.nb_trajectory = 0
		self.threshold_mask = []
		self.distance_matrix = []
		
	def get_trajectories(self,pixels = 1):
		if len(self.trajectories) == 0:
			database = sqlite3.connect(self.filename)
			cursor = database.cursor()
			
			db_request1 = "SELECT max(trajectory_id), min(trajectory_id) FROM positions"
			max_min = cursor.execute(db_request1).fetchall()
			nb_trajectory = max_min[0][0]- max_min[0][1] +1
			
			trajectories = [cursor.execute(
							"SELECT p.x_coordinate,p.y_coordinate \
							FROM positions as p \
							WHERE trajectory_id ="+str(i)+" ORDER BY p.frame_number ASC"
						).fetchall() for i in range(nb_trajectory)]

			self.trajectories = np.array([ Trajectory(e) for e in trajectories ])
			database.close()
			self.nb_trajectory = len(self.trajectories)
			self.threshold_mask = np.arange(self.nb_trajectory, dtype = int).tolist()
		
		if pixels == 1:
			for trajectory in self.trajectories:
				trajectory.to_pixel(self.homography)
		
		

	def discretize_pixel_trajectories(self,square_side_size = 50):
		for trajectory in self.trajectories:
			trajectory.discretize(square_side_size)
	
	
	
	
	
	# TODO mask if discretized
	def compute_dtw_distance_matrix(self, distance = cityblock,mask = 0):
		masked_pixel_trajectories = [self.trajectories[i].discretized_trajectory for i in self.threshold_mask ]
		l = len(masked_pixel_trajectories)
		print(l)
		D = np.zeros((l,l))
		for i in range(l):
			for j in range(i,l):
				#D[i,j] = my_affinity_wrapper(trajectories[i],trajectories[j])
				#D[i,j],p = fastdtw(trajectories[i],trajectories[j], dist=euclidean)
				D[i,j],p = fastdtw(masked_pixel_trajectories[i],masked_pixel_trajectories[j], dist = distance)
				D[j,i] = D[i,j]
		self.distance_matrix = D
	
	# TODO how to know what where the parmeters used for computing this
	# TODO implement file not found case
	def load_dtw_distance_matrix(self,filename):
		self.distance_matrix = np.loadtxt(filename)
		
		
	def save_dtw_distance_matrix(self,filename):
		np.savetxt(self.distance_matrix, filename)

	## Extracted features

	def extract_features(self):
		for trajectory in self.trajectories:
			trajectory.compute_features()

	def standardize_features(self):
		std_data = pd.DataFrame([trajectory.extracted_features for trajectory in self.trajectories])
		scaler = StandardScaler()
		std_data[std_data.columns] = scaler.fit_transform(std_data[std_data.columns]) 
		std_data = std_data.as_matrix()
		for i in range(len(std_data)):
			self.trajectories[i].standardized_features = std_data[i]

	def display_correlation_matrix(self):
		std_data = pd.DataFrame([trajectory.extracted_features for trajectory in self.trajectories])
		corr_matrix = std_data.corr()
		mask = np.zeros_like(corr_matrix, dtype=np.bool)
		mask[np.triu_indices_from(mask)] = True


		f, ax = plt.subplots(figsize=(11, 9))

		cmap = sns.diverging_palette(220, 10, as_cmap=True)

		# Draw the heatmap with the mask and correct aspect ratio
		sns.heatmap(corr_matrix,mask = mask, cmap=cmap, vmax=.3, center=0,
					square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Distance between first and last point

# Number of points in the trajectory

# Trajectory_x min, 25 , mean , 75 , max

# Trajectory_y min, 25 , mean , 75 , max

# Velocity min, 25 , mean , 75 , max

# Rate of turn min, 25 , mean , 75 , max

# Difference between two consecutives Trajectory_x min, 25 , mean , 75 , max

# Difference between two consecutives Trajectory_y min, 25 , mean , 75 , max

# Difference between two consecutives Velocity min, 25 , mean , 75 , max

# Difference between two consecutives Rate of turn min, 25 , mean , 75 , max