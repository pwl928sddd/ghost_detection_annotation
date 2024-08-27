from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

class myDataset(Dataset):

	def __init__(self, root_dir, 
				 split='train',
				 transform=None):
		self.root = root_dir
		self.npoints = 2500

		splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
		self.filelist = json.load(open(splitfile, 'r'))

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, idx):
		point_set = np.loadtxt(self.root + self.filelist[idx], skiprows=1, delimiter=" ", usecols=[0, 1, 2, 3, 4, 5])
		point_set = point_set[np.where((point_set[:,0]<50) * (point_set[:,1]<50) * (point_set[:,2]<50))]
		filepath = self.root + self.filelist[idx]
		num_points = 0
		if(len(point_set) >= self.npoints):
			choice = np.random.choice(len(point_set), self.npoints, replace=True)
			point_set = point_set[choice, :]
			num_points = self.npoints
		else:
			zeros_length = self.npoints - len(point_set)
			zeros = np.zeros([zeros_length, 6])
			num_points = len(point_set)
			point_set = np.concatenate((point_set, zeros))


		data = point_set[:, 0:5]
		label = point_set[:, 5]
		num_pts = num_points

		return data, label, num_pts, filepath

if __name__ == '__main__':
	batch_size = 32

	dataset = myDataset(root_dir='/home/mahapatro/pointnet.pytorch/custom_data/', split="val")
	train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
	for i, data in enumerate(train_loader, 0):
		points, target = data
		print(points[0], target[0])
