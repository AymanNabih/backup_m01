import sqlite3
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean,cityblock,squareform
from fastdtw import fastdtw

class Trajectory:
	def __init__(self,trajectory):
		self.meter_trajectory = trajectory
		self.pixel_trajectory = []
		self.discretized_trajectory = []
		self.original_length = len(trajectory)
		self.pixel = False
		self.discretized = 0
		
	def to_pixel(self,homography):
		if not self.pixel:
			for point in self.meter_trajectory:
				new_point = np.array([point[0],point[1],1])
				new_point = np.matmul(homography,new_point)
				new_point = [new_point[0]/new_point[2],new_point[1]/new_point[2]]
				new_points.append(new_point)
			pixel_points.append(new_points)
			self.pixel_trajectory = pixel_points
			self.pixel = True
		
	def discretize(self,square_side_size = 50):
		if self.discretized != 0 and square_side_size != self.discretized:
			discretized_trajectory = []
			for point in self.pixel_trajectory:
				i = int(point[1]/square_side_size)
				j = int(point[0]/square_side_size)
				square = [i,j]
				if  len(discretized_trajectory) == 0 or discretized_trajectory[-1] != square:
					discretized_trajectory.append(square)
			self.discretized_trajectory = discretized_trajectory
			self.discretized = square_side_size

		
		
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
			
			self.trajectories = np.array([
				Trajectory(
						cursor.execute(
							"SELECT p.x_coordinate,p.y_coordinate \
							FROM positions as p \
							WHERE trajectory_id ="+str(i)+" ORDER BY p.frame_number ASC"
						).fetchall() for i in range(nb_trajectory)
					)
				])
			database.close()
		self.nb_trajectory = len(self.trajectories)
		self.threshold_mask = np.arange(self.nb_trajectory, dtype = int).tolist()
		
		if pixels == 1:
			for trajectory in self.trajectories:
				trajectory.to_pixel(self.homography)
		
		

	def discretize_pixel_trajectories(self,square_side_size = 50):
		for trajectory in self.trajectories:
			trajectory.discretize(square_side_size)
	
	
	
	def plot_trajectory_lengths_distribution(self,nb_bins = 20, mask = 0):
		lengths = []
		if pixels:
			if mask:
				lengths = [trajectory.original_length for trajectory in [self.trajectories[i] for i in self.threshold_mask ]]
				#lengths = [len(t) for t in self.pixel_trajectories[self.threshold_mask]]
			else:
				lengths = [original_length for trajectory in self.trajectories]
		plt.hist(lengths,bins = nb_bins)
		plt.show()
		
		
		
	def threshold_pixels_trajectory_length(self,threshold = 40 ):
		self.threshold_mask = [i for i,trajectory in enumerate(self.trajectories) if trajectory.original_length > threshold]
	
	
	
	def compute_dtw_distance_matrix(self, distance = cityblock):
		masked_pixel_trajectories = [self.trajectories[i].pixel_trajectory for i in self.threshold_mask ]
		l = len(masked_pixel_trajectories)
		D = np.zeros((l,l))
		for i in range(l):
			for j in range(i,l):
				#D[i,j] = my_affinity_wrapper(trajectories[i],trajectories[j])
				#D[i,j],p = fastdtw(trajectories[i],trajectories[j], dist=euclidean)
				D[i,j],p = fastdtw(masked_pixel_trajectories[i],masked_pixel_trajectories[j], dist = distance)
				D[j,i] = D[i,j]
		self.distance_matrix = D
	
	
	def load_dtw_distance_matrix(self,filename):
		self.distance_matrix = np.loadtxt(filename)
		
		
	def save_dtw_distance_matrix(self,filename):
		np.savetxt(self.distance_matrix, filename)
		
		
		
#def meters_to_pixels(self, meter_trajectories):
	#	pixel_points = []
	#	for meter_trajectory in meter_trajectories:
	#		new_points = []
	#		for point in meter_trajectory:
	#			new_point = np.array([point[0],point[1],1])
	#			new_point = np.matmul(self.homography,new_point)
	#			new_point = [new_point[0]/new_point[2],new_point[1]/new_point[2]]
	#			new_points.append(new_point)
	#		pixel_points.append(new_points)
	#	return pixel_points
	
#def discretize(self,trajectory,square_side_size = 50 ):
#		Discretized_trajectory = []
#		for point in trajectory:
#			i = int(point[1]/square_side_size)
#			j = int(point[0]/square_side_size)
#			square = [i,j]
#			if  len(Discretized_trajectory) == 0 or Discretized_trajectory[-1] != square:
#				Discretized_trajectory.append(square)
#		return Discretized_trajectory