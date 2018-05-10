import torch.nn.functional as F
from torch import nn

class PowerNet(nn.Module):

	def __init__(self, input_size, output_size, initial_size=256):
		super(PowerNet, self).__init__()

		self.input_size = input_size,
		self.output_size = output_size
		self.initial_size = initial_size
		self.reservoir_size = initial_size


		self.fc_in = nn.Linear(in_features=input_size,
							   out_features=initial_size)

		self.fc_out = nn.Linear(in_features=initial_size,
								out_features=output_size)

	def forward(self, x):

		x = self.fc_in(x)

		# pad x with zeros 

	def grow_network(self, n_new_nodes=1):

		self.reservoir_size += n_new_nodes
		


