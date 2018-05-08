import sys
import os

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F 
from torch.autograd import Variable

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from powernet import PowerNet

use_cuda = torch.cuda.is_available()
root = './data'
# download MNIST dataset or not
if os.path.isdir(root + '/raw'):
	download = False 
else:
	download = True

def train(model, dataset, n_epochs=5, learning_rate=1e-3, batch_size=16):

	if use_cuda:
		model = model.cuda()

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()


	for epoch in range(n_epochs):

		for inputs, targets in dataset['train']:

			model.zero_grad()

			inputs = inputs.view(batch_size, -1)
			inputs, targets = Variable(inputs), Variable(targets)
			if use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda()

			outputs = model(inputs, n_steps=3)

			loss = criterion(outputs, targets)

			loss.backward()
			optimizer.step()

		# printing results

		print("Epoch : %d ----- Loss : %.3f")
	


if __name__ == "__main__":

	batch_size = 16

	input_size = 784
	reservoir_size = 256
	output_size = 10

	trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
	train_set = MNIST(root=root, train=True, transform=trans, download=download)
	test_set = MNIST(root=root, train=False, transform=trans)

	dataloaders ={}
	dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=True)
	dataloaders['test'] = DataLoader(test_set, batch_size=batch_size, shuffle=False)

	model = PowerNet(input_size=input_size,
					 output_size=output_size,
					 reservoir_size=reservoir_size)

	train(model, dataloaders, batch_size=batch_size)