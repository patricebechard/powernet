import sys
import os
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from powernet import PowerNet
from evaluate import evaluate

use_cuda = torch.cuda.is_available()
root = './data'
results_root = './results/'
# download MNIST dataset or not
if os.path.isdir(root + '/raw'):
	download = False 
else:
	download = True

def train(model, dataset, n_epochs=20, learning_rate=1e-3, batch_size=16, n_steps=3):

	if use_cuda:
		model = model.cuda()

	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()

	loss_buffer = []
	train_acc_buffer = []
	test_acc_buffer = []

	for epoch in range(n_epochs):

		epoch_loss = 0
		n_batches = len(dataset['train'])

		total_good = 0.
		n_ex = 0.

		for inputs, targets in dataset['train']:

			model.zero_grad()

			inputs = inputs.view(batch_size, -1)
			inputs, targets = Variable(inputs), Variable(targets)
			if use_cuda:
			    inputs, targets = inputs.cuda(), targets.cuda()

			outputs = model(inputs, n_steps=n_steps)

			loss = criterion(outputs, targets)
			epoch_loss += loss.data[0]

			loss.backward()
			optimizer.step()

			_, predictions = torch.max(outputs, dim=1)
			total_good += torch.sum(torch.eq(predictions, targets)).data[0]
			n_ex += len(targets)

		# printing results
		epoch_loss /= n_batches
		train_acc = (total_good / n_ex) * 100
		test_acc = evaluate(model, dataset)

		print("Epoch : %02d ----- Loss : %.3f ----- Train acc : %.2f%% ----- Test acc : %.2f%%" % (epoch, 
																								 epoch_loss,
																								 train_acc,
																								 test_acc))

		loss_buffer.append(epoch_loss)
		train_acc_buffer.append(train_acc)
		test_acc_buffer.append(test_acc)

	loss_buffer = np.array(loss_buffer)
	train_acc_buffer = np.array(train_acc_buffer)
	test_acc_buffer = np.array(test_acc_buffer)

	np.savetxt(results_root + 'loss_%02d.txt' % n_steps, loss_buffer)
	np.savetxt(results_root + 'train_acc_%02d.txt' % n_steps, train_acc_buffer)
	np.savetxt(results_root + 'test_acc_%02d.txt' % n_steps, test_acc_buffer)

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

	for n_steps in range(10):
		model = PowerNet(input_size=input_size,
						 output_size=output_size,
						 reservoir_size=reservoir_size)

		train(model, dataloaders, batch_size=batch_size, n_steps=n_steps)